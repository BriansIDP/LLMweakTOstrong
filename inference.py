import os, sys
import re
import random
import time
import json
import argparse
import math

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteriaList, StoppingCriteria
from peft import PeftModel, PeftConfig

from data.prompt import templates, prompts
from knowledgemodel import KnowledgeLLM
from FSA import LexFSA
# from model import KnowledgeLLM
from scoring.evaluation.metrics import ErrorMetric
from scoring.evaluation.util import format_results, load_predictions, load_gold_data
from scoring.evaluation.normalizers.english import EnglishTextNormalizer

normaliser = EnglishTextNormalizer()


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
random.seed(1)
torch.manual_seed(1)


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = '</s>', tokenizer=None):
        self.stops = stops
        self.tokenizer = tokenizer
        StoppingCriteria.__init__(self),

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return self.tokenizer.decode(input_ids[0, -5:]).endswith(self.stops)


def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

def get_knowledge_index(knowledge):
        knowledge_index = {}
        with open(knowledge) as fin:
            data = json.load(fin)
            for key, values in data.items():
                for value in values:
                    knowledge_str = "{} is a type of {}".format(value, key)
                    if value not in knowledge_index:
                        knowledge_index[value] = []
                    knowledge_index[value].append(knowledge_str)
        return knowledge_index

def gather_knowledge(utterance, knowledge, maxKBsize=0, linearise_knowledge=False):
    sel_knowledge = {"<slot>": [], "<value>": {}}
    count = 0
    self_knowledge_str = []
    for value, content in knowledge.items():
        inutt = False
        uttlist = utterance.split()
        value = value.split()
        for i in range(len(uttlist)):
            if value == uttlist[i:i+len(value)]:
                inutt = True
        if inutt:
            self_knowledge_str.extend(content)
            for kitem in content:
                value, slot = kitem.split(" is a type of ")
                if slot not in sel_knowledge["<slot>"]:
                    sel_knowledge["<slot>"].append(slot)
                    sel_knowledge["<value>"][slot] = [value]
                else:
                    sel_knowledge["<value>"][slot].append(value)
    return sel_knowledge, json.dumps(sel_knowledge["<value>"]) # ", ".join(self_knowledge_str) # json.dumps(sel_knowledge["<value>"])

def loadknowledge(knowledge_lki, orig_lki):
    knowledgedict = {}
    for knowledge in knowledge_lki:
        try:
            kitem = json.loads(knowledge)
        except:
            kitem = {}
        if not isinstance(kitem, dict):
            kitem = {}
        for key, values in kitem.items():
            values = values.split(" & ") if isinstance(values, str) else values
            for value in values:
                if key in orig_lki["<value>"]: # and value in orig_lki["<value>"][key]:
                    if key in knowledgedict:
                        knowledgedict[key].append(value)
                    else:
                        knowledgedict[key] = [value]
    return json.dumps(knowledgedict)

def merge_outputs(outputs, slotdict):
    new_output = {}
    for output in outputs:
        if "</s>" in output:
            output = output.split("</s>")[0]
        try:
            outdict = json.loads(output)
            for key, value in outdict.items():
                value = " & ".join(value) if isinstance(value, list) else value
                if value not in [i for v in new_output.values() for i in v]:
                    if key in new_output and value not in new_output[key]:
                        new_output[key].append(value)
                    elif key not in new_output and key in slotdict:
                        new_output[key] = [value]
        except:
            continue
    for key, value in new_output.items():
        new_output[key] = " & ".join(value)
    return json.dumps(new_output)

def adapter_ensemble(model, inputs, generated, tokenizer, nadapters):
    yseqs = [torch.tensor(g.yseq).to(device) for g in generated]
    input_part = pad_sequence(yseqs, batch_first=True, padding_value=0)
    labels = pad_sequence(yseqs, batch_first=True, padding_value=-1)
    total_inputs = torch.cat([inputs.repeat(labels.size(0), 1), input_part], dim=-1)[:, :-1]
    total_labels = torch.cat([inputs.repeat(labels.size(0), 1) * 0 - 1, labels], dim=-1)[:, 1:]
    attn_mask = total_inputs != 0
    inputs_bundle = {"input_ids": total_inputs, "attention_mask": attn_mask}
    forward_probs = []
    label_mask = labels != -1
    labels = labels * label_mask
    for i in range(nadapters):
        model.llm.set_adapter("ada_{}".format(i+1))
        forward_logits = model(inputs_bundle, total_labels)[0].logits[:, inputs.size(1)-1:]
        if i == 0:
            forward_logp = torch.softmax(forward_logits, dim=-1)
        else:
            forward_logp += torch.softmax(forward_logits, dim=-1)
    forward_logp = torch.log(forward_logp / nadapters).view(-1, forward_logp.size(-1))
    forward_entropy = (- torch.exp(forward_logp) * forward_logp).sum(dim=-1)
    forward_entropy = forward_entropy.view(labels.size(0), -1)
    forward_entropy = (forward_entropy * label_mask).tolist()
    forward_logp = forward_logp[torch.arange(forward_logp.size(0)), labels.view(-1)].view(labels.size(0), -1)
    # forward_entropy = (forward_logp * label_mask).tolist()
    # Product of expectation
    forward_logp = (forward_logp * label_mask).sum(dim=-1)
    model.llm.set_adapter("ada_1")
    return forward_logp, forward_entropy

def get_cascaded_uncertainty(model, prompt_nbest, generate_hyps, tokenizer, lengths):
    T = 0.3 # 0.001
    inputs = []
    labels = []
    asr_scores = []
    startpos = []
    nasrhyps = len(prompt_nbest)
    nhyps = len(generate_hyps)
    for asr_hyp in prompt_nbest:
        asr_scores.append(asr_hyp[1])
        startpos.append(len(asr_hyp[0]))
        for hyp in generate_hyps:
            local_input = torch.tensor(asr_hyp[0] + hyp.yseq).to(device)
            local_label = torch.tensor([-1] * len(asr_hyp[0]) + hyp.yseq).to(device)
            inputs.append(local_input)
            labels.append(local_label)

    # Get ASR output distribution
    asr_dist = torch.softmax(torch.tensor(asr_scores).to(device) / T, dim=-1)

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)[:, :-1]
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)[:, 1:]
    attn_mask = inputs != 0
    label_mask = labels != -1
    inputs_bundle = {"input_ids": inputs, "attention_mask": attn_mask}

    # Forward model
    forward_logits = model(inputs_bundle, labels)[0].logits #.view(nasrhyps, nhyps, inputs.size(1), -1)
    forward_logp = torch.log_softmax(forward_logits, dim=-1)
    # forward_entropy = (- torch.exp(forward_logp) * forward_logp).sum(dim=-1).view(nasrhyps, nhyps, -1)

    forward_logp = forward_logp.view(-1, forward_logp.size(-1))
    forward_logp = forward_logp[torch.arange(forward_logp.size(0)), labels.reshape(-1)].reshape(labels.size(0), -1)

    forward_entropy = (forward_logp * label_mask).view(nasrhyps, nhyps, -1)
    merged_entropy = []
    for i, pos in enumerate(startpos):
        merged_entropy.extend([ent[pos-1:] for ent in forward_entropy[i]])
    merged_entropy = pad_sequence(merged_entropy, batch_first=True, padding_value=0).view(nasrhyps, nhyps, -1)
    merged_entropy = (merged_entropy * asr_dist.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    seq_logp = (forward_logp * label_mask).sum(dim=-1)
    seq_logp = seq_logp.view(nasrhyps, nhyps)
    cascaded_entropies = []
    entropy, unnorm_entropy, _ = calc_predictive_entropy(seq_logp, 1.0, lengths.unsqueeze(0))
    entropy = (entropy * asr_dist).sum(dim=0)
    unnorm_entropy = (unnorm_entropy * asr_dist).sum(0)
    return merged_entropy.tolist(), entropy, unnorm_entropy


def calc_predictive_entropy(logp, temperature, lengths):
    pi_b = torch.softmax(logp / temperature, dim=-1)
    entropy = - (logp * pi_b / lengths).sum(dim=-1)
    entropy_seq = - logp * pi_b / lengths
    entropy_unnorm = - (logp * pi_b).sum(dim=-1)
    return entropy, entropy_unnorm, entropy_seq

def segment_uncertainty(entropy_seq, output_ids, tokenizer):
    slot_value_pairs = []
    for i, output in enumerate(output_ids):
        cumout = []
        cumentropy = []
        if tokenizer.decode(output[:-1]) == "{}":
            slot_value_pairs.append(({}, sum(entropy_seq[i][:1])))
        else:
            for k, ids in enumerate(output[:-1]):
                if tokenizer.decode(cumout + [ids]).endswith(",") or tokenizer.decode(cumout + [ids]).endswith("}"):
                    segstr = tokenizer.decode(cumout)
                    if segstr.startswith("{"):
                        segstr = segstr[1:]
                    if not segstr.endswith(","):
                        segstr = segstr + "\""
                    try:
                        slotvalue = json.loads("{"+segstr+"}")
                        slot_value_pairs.append((slotvalue, sum(cumentropy)/len(cumentropy)))
                    except:
                        pass
                    cumout = []
                    cumentropy = []
                else:
                    cumout.append(ids)
                    cumentropy.append(entropy_seq[i][k])
    return slot_value_pairs

def calc_metrics(output, label):
    span_f1 = ErrorMetric.get_instance(metric="span_f1", average="micro")
    distance_metrics = {}
    for distance in ['word', 'char']:
        distance_metrics[distance] = ErrorMetric.get_instance(metric="span_distance_f1",
                                                              average="micro",
                                                              distance=distance)
    slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average="micro")
    try:
        output = json.loads(output)
        output_format = []
        for key, value in output.items():
            output_format.append({"type": key, "filler": normaliser(value).replace(" 's", "'s")})
    except:
        output_format = []
    label_format = []
    for key, value in label.items():
        label_format.append({"type": key, "filler": value})
    span_f1(label_format, output_format)
    span_results = span_f1.get_metric()
    for distance, metric in distance_metrics.items():
        metric(label_format, output_format)
        results = metric.get_metric()
        slu_f1(results)
    slu_f1 = slu_f1.get_metric()
    if output_format == [] and label_format == []:
        return 1, 1
    else:
        return span_results["overall"][2], slu_f1["overall"][2]


def calc_segment_metrics(uttlabel, segment_output):
    correctness = []
    uncertainties = []
    for pair in segment_output:
        slotvalue, uncertainty = pair
        if uncertainty < 0:
            uncertainty = 1 - math.exp(uncertainty)
        hit = 1
        if slotvalue == {}:
            if uttlabel != {}:
                hit = 0
        else:
            for slot, value in slotvalue.items():
                if slot not in uttlabel or uttlabel[slot] != value:
                    hit = 0
        correctness.append(hit)
        uncertainties.append(uncertainty)
    return correctness, uncertainties


def main(args):
    start = time.time()
    with open(os.path.join(args.model_path, "model_config.json")) as fin:
        train_args = json.load(fin)
    checkpoints = []
    if args.ckptlist != "":
        with open(args.ckptlist) as fin:
            checkpoints = [line.strip() for line in fin]
    ## Meta data
    with open("data/ontology_norm.json") as fin:
        knowledgebase = json.load(fin)
    with open("data/slotlist.json") as fin:
        slotdict = json.load(fin)
        slotstr = ", ".join(['"' + key + '"' for key in slotdict.keys()])
    candidates = getattr(train_args, "num_candidates", 1)

    # Load knowledge
    knowledge_index = None
    knowledge_embeds = None
    if args.ontology != "":
        knowledge_index = get_knowledge_index(args.ontology)
    linearise_knowledge = "LKI" in train_args["tag"] or "LKI" in args.tag

    ## Initialise tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_args["model_path"])
    weak_tokenizer = AutoTokenizer.from_pretrained(train_args["weak_model_path"])

    # Stopping criterion
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops='</s>', tokenizer=tokenizer)])

    # determine model type
    LLMtype = "vicuna"
    if "llama-2" in train_args["model_path"]:
        LLMtype = "llama2"

    # Load model checkpoint
    llm = AutoModelForCausalLM.from_pretrained(
        train_args["model_path"],
        torch_dtype=torch.float16 if "gpt2" not in train_args["model_path"] else torch.float32,
        cache_dir="/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMknowledge/cache",
    )
    weakllm = AutoModelForCausalLM.from_pretrained(
        train_args["weak_model_path"],
        torch_dtype=torch.float16 if "gpt2" not in train_args["model_path"] else torch.float32,
        cache_dir="/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMknowledge/cache",
    )
    if args.ckptlist != "":
        for k, ckpt in enumerate(checkpoints):
            if k == 0:
                llm = PeftModel.from_pretrained(llm, ckpt, adapter_name="ada_1")
            else:
                llm.load_adapter(ckpt, adapter_name="ada_{}".format(k+1))
        llm.set_adapter("ada_1")
    elif "gpt2" not in train_args["model_path"]:
        # config = PeftConfig.from_pretrained(peftpath)
        llm = PeftModel.from_pretrained(llm, os.path.join(args.model_path, args.main_ckpt), adapter_name="ada_1")
    else:
        state_dict = torch.load(os.path.join(args.model_path, args.main_ckpt, "pytorch_model.pt"))
        llm.load_state_dict(state_dict)

    if "gpt2" not in train_args["weak_model_path"]:
        # config = PeftConfig.from_pretrained(peftpath)
        llm.load_adapter(os.path.join(args.model_path, "checkpoint.best_weak"), adapter_name="ada_weak")
        llm.set_adapter("ada_1")
    else:
        state_dict = torch.load(os.path.join(args.model_path, "checkpoint.best_weak", "pytorch_model.pt"))
        weakllm.load_state_dict(state_dict)
        weakllm.to(device)

    model = KnowledgeLLM(
        llm,
        tokenizer,
        train_args["use_lora"],
    )
    model = model.to(device)
    model.eval()

    # Read test file
    with open(args.recogfile) as fin:
        utterances = json.load(fin)

    # Global knowledge
    sel_knowledge = {"<slot>": list(knowledgebase.keys()), "<value>": knowledgebase}
    values = LexFSA(tokenizer, task="slot", groupliteral=sel_knowledge)

    start = time.time()
    outputdict = {}
    count = 0

    with torch.no_grad():
        for slurpid, utterance in utterances.items():
            system = prompts["system"]
            taskdesc = prompts["task_description"].format(slotstr)
            query = prompts["query"] if args.topn <= 1 else prompts["user2nbest"]
            knowledge = None
            if isinstance(utterance, dict):
                uttdict = utterance
                if "score" in uttdict:
                    nbest_utts = sorted(zip(uttdict["nbest"], uttdict["score"]), key=lambda tup: tup[1], reverse=True)
                    nbest_utt_str = [text[0] for text in nbest_utts]
                    utterance = nbest_utts[0][0] if args.topn <= 1 else "\n".join([hyp[0] for hyp in nbest_utts[:args.topn]])
                else:
                    utterance = uttdict["nbest"][0]
            elif isinstance(utterance, list):
                utterance = utterance[5] if args.topn <= 1 else "\n".join(utterance[:args.topn])

            if "nbest" in args.tag:
                nbest = "\n".join(uttdict["nbest"])
                knowledge, knowledge_lki = gather_knowledge(nbest, knowledge_index, args.maxKBsize, linearise_knowledge)
            else:
                knowledge, knowledge_lki = gather_knowledge(utterance, knowledge_index, args.maxKBsize, linearise_knowledge)
            values = LexFSA(tokenizer, task="slot", groupliteral=knowledge)

            if "upperbound" in args.tag:
                knowledge_lki = uttdict["label"]
                utterance = uttdict["text"]
            content = utterance

            # Forward first pass
            prompt = templates[LLMtype]["slot"][1].format(**locals())
            if "gpt2" not in train_args["weak_model_path"]:
                model.llm.set_adapter("ada_weak")
            else:
                model.llm = weakllm
                model.tokenizer = weak_tokenizer
            inputs = weak_tokenizer(prompt, return_tensors="pt").to(device)
            generate_hyps = model.generate_beam(
                input_ids=inputs.input_ids,
                max_new_tokens=60,
                stopping_criteria=stopping_criteria,
                beamsize=candidates,
                n_adapters=1,
            )
            weak_output = [weak_tokenizer.decode(hyp.yseq).split("</s>")[0] for hyp in generate_hyps]
            # Calc uncertainty
            lengths = torch.tensor([len(hyp.yseq) for hyp in generate_hyps]).to(device)
            logplist = torch.stack([hyp.cumscore for hyp in generate_hyps])
            logplist = (- logplist / lengths).tolist()
            if args.unc_threshold >= 0:
                for i, weak_out in enumerate(weak_output):
                    weak_output[i] = weak_output[i] + "<uncertainty: {:.2f}>".format(logplist[i])
            origquery = query
            query = prompts["delib_query"].format(", ".join(weak_output)) + query
            if "gpt2" not in train_args["weak_model_path"]:
                model.llm.set_adapter("ada_1")
            else:
                model.llm = llm
                model.tokenizer = tokenizer

            # Do second pass now
            for i in range(args.iterations):
                if linearise_knowledge:
                    prompt = templates[LLMtype]["slot"][1].format(**locals())
                else:
                    prompt = templates[LLMtype]["slot"][0].format(**locals()) 
                    if args.cascaded:
                        prompt_nbest = []
                        for k, utt in enumerate(nbest_utt_str):
                            content = utt
                            prompt_nbest.append((tokenizer(templates[LLMtype]["slot"][0].format(**locals()))["input_ids"], nbest_utts[k][1]))
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                generate_hyps = model.generate_beam(
                    input_ids=inputs.input_ids,
                    max_new_tokens=60,
                    stopping_criteria=stopping_criteria,
                    knowledge=knowledge_embeds if not linearise_knowledge else None,
                    values=values if "select" not in args.tag else None,
                    beamsize=candidates,
                    n_adapters=1, # len(checkpoints),
                )
                output = [tokenizer.decode(hyp.yseq).split("</s>")[0] for hyp in generate_hyps]
                query = prompts["delib_query"].format(", ".join(output)) + origquery

            # Uncertainty
            newbest_ind = 0
            lengths = torch.tensor([len(hyp.yseq) for hyp in generate_hyps]).to(device)
            if args.cascaded:
                ensemble_entropy, predictive_entropy, unnorm_entropy  = get_cascaded_uncertainty(
                    model, prompt_nbest[:5], generate_hyps, tokenizer, lengths)
                logplist = torch.stack([hyp.cumscore for hyp in generate_hyps])
                newbest_ind = torch.sort(logplist / lengths, descending=True)[1][0]
            else:
                if len(checkpoints) <= 1:
                    logplist = torch.stack([hyp.cumscore for hyp in generate_hyps])
                    ensemble_entropy = [torch.stack(hyp.entropy).tolist() for hyp in generate_hyps]
                    # ensemble_entropy = [hyp.scores for hyp in generate_hyps]
                else:
                    logplist, ensemble_entropy = adapter_ensemble(
                        model, inputs.input_ids, generate_hyps, tokenizer, len(checkpoints))
                newbest_ind = torch.sort(logplist / lengths, descending=True)[1][0]
                predictive_entropy, unnorm_entropy, _ = calc_predictive_entropy(logplist, args.calibration_t, lengths)
            cutoff = 1
            if args.do_sampling:
                results = monte_carlo_dropout(model, inputs.input_ids, generate_hyps, tokenizer)
                predictive_entropy, unnorm_entropy, _ = calc_predictive_entropy(results[0].sum(dim=-1), args.calibration_t, lengths)
            # Measure segment level uncertainty
            segment_output = segment_uncertainty([ensemble_entropy[newbest_ind]], [generate_hyps[newbest_ind].yseq], tokenizer)
            correctness, uncertainty = calc_segment_metrics(uttdict["label"], segment_output)
            bs_entropy = torch.stack(generate_hyps[newbest_ind].entropy).sum() / len(generate_hyps[newbest_ind].yseq)

            # Get outputs
            outputs = tokenizer.batch_decode([generate_hyps[newbest_ind].yseq], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            output = merge_outputs(outputs, slotdict)
            print(predictive_entropy, unnorm_entropy, output)
            entity_f1, slu_f1 = calc_metrics(output, uttdict["label"])

            outputdict[slurpid] = {
                "output": output,
                "predictive": predictive_entropy.item(),
                "unnormalised": unnorm_entropy.item(),
                "entity_f1": entity_f1,
                "slu_f1": slu_f1,
                "beamsearch_entropy": bs_entropy.item(),
                "seg_correctness": correctness,
                "seg_uncertainty": uncertainty,
            }
            count += 1
            logging("Finished {}, Elapsed time {:.2f}".format(count, time.time()-start))

    with open(os.path.join(args.model_path, "output_{}_top{}{}.json".format(args.asrname, args.topn, args.tag)), "w") as fout:
        json.dump(outputdict, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM finetuning")
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the model file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama13b",
        help="model name",
    )
    parser.add_argument(
        "--recogfile",
        type=str,
        default="dataset/gt_nbest_sel.json",
        help="Path to the model file",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="log.output",
        help="Path to the model file",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=1,
        help="model name",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="model name",
    )
    parser.add_argument(
        "--asrname",
        type=str,
        default="gt",
        help="model name",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="model name",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default="",
        help="KB for biasing",
    )
    parser.add_argument(
        "--knowledge_embs",
        type=str,
        default="",
        help="Pre-computed knowledge embedding",
    )
    parser.add_argument(
        "--maxKBsize",
        type=int,
        default=10,
        help="Size of the biasing list to use",
    )
    parser.add_argument(
        "--cutoff_prob",
        type=float,
        default=0.0,
        help="Top P probability",
    )
    parser.add_argument(
        "--ckptlist",
        type=str,
        default="",
        help="List of checkpoints for ensemble estimation",
    )
    parser.add_argument(
        "--main_ckpt",
        type=str,
        default="",
        help="main checkpoint path",
    )
    parser.add_argument(
        "--do_sampling",
        action='store_true',
        help="Use sampling for uncertainty estimation",
    )
    parser.add_argument(
        "--calibration_t",
        type=float,
        default=1.0,
        help="Calibration temperature",
    )
    parser.add_argument(
        "--cascaded",
        action='store_true',
        help="Compute cascaded uncertainty" 
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of deliberation iterations"
    )
    parser.add_argument(
        "--unc_threshold",
        type=float,
        default=10000.0,
        help="Portion to be considered as uncertain",
    )
    args = parser.parse_args()
    main(args)
