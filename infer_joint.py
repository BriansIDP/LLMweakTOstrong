# import debugpy

# # 5678是debugpy服务器监听的端口号，确保这个端口在你的系统上是空闲的
# debugpy.listen(('0.0.0.0', 5678))
# print("⏳ Waiting for debugger to attach...")

# # 让debugpy等待VSCode的调试器连接
# debugpy.wait_for_client()
# print("🚀 Debugger attached!")


import os
import random
import time
import json
import argparse
import math
import string

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteriaList, StoppingCriteria
from transformers.modeling_utils import load_sharded_checkpoint, safe_load_file, load_state_dict
from peft import PeftModel, PeftConfig

from data.prompt import templates, prompts
from knowledgemodel import KnowledgeLLM
# from model import KnowledgeLLM
from scoring.evaluation.metrics import ErrorMetric
from scoring.evaluation.util import format_results, load_predictions, load_gold_data
from scoring.evaluation.normalizers.english import EnglishTextNormalizer

normaliser = EnglishTextNormalizer()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def merge_outputs(output, slotdict):
    new_output = {}
    # for output in outputs:
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
        return "{}"
    for key, value in new_output.items():
        try:
            new_output[key] = " & ".join(value)
        except:
            continue
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
    start_time = time.time()
    with open(os.path.join(args.model_path, "model_config.json")) as fin:
        train_args = json.load(fin)
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

    # ## Initialise tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(train_args["model_path"], use_fast=("pythia" in train_args["model_path"] or "bloom" in train_args["model_path"]))

    # Stopping criterion
    # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops='</s>', tokenizer=tokenizer)])

    # determine model type
    LLMtype = "vicuna"
    # if "llama-2" in train_args["model_path"]:
    #     LLMtype = "llama2"

    weak_model_names = ["gpt2-large", "opt-1.3b", "pythia-1.4b"]
    model_list = []
    tokenizer_list = []
    for weak_model_name in weak_model_names:
        pretrained_weak_model_path = os.path.join("exp/weak", weak_model_name, 'checkpoint.best')
        weak_model_path = os.path.join("/mnt/nvme_share/cuizy/models", weak_model_name)
        weak_tokenizer = AutoTokenizer.from_pretrained(weak_model_path, use_fast=("pythia" in weak_model_path), trust_remote_code=True)
        if os.path.exists(pretrained_weak_model_path):
            if os.path.exists(os.path.join(pretrained_weak_model_path, "model.safetensors")):
                weakllm = AutoModelForCausalLM.from_pretrained(
                    os.path.join(pretrained_weak_model_path),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                    device_map="auto",
                )
            else:
                weakllm = AutoModelForCausalLM.from_pretrained(
                    weak_model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                    device_map="auto",
                )
                if os.path.exists(os.path.join(pretrained_weak_model_path, "pytorch_model.pt")):
                    state_dict = torch.load(os.path.join(pretrained_weak_model_path, "pytorch_model.pt"))
                    weakllm.load_state_dict(state_dict)
                else:
                    load_sharded_checkpoint(weakllm, os.path.join(pretrained_weak_model_path))
            weakmodel = KnowledgeLLM(weakllm, weak_tokenizer)
            weakmodel.eval()
        else:
            print("Error: Please input correct pretrained weak model path.")
            return 0
        model_list.append(weakmodel)
        tokenizer_list.append(weak_tokenizer)

    # Read test file
    with open(args.recogfile) as fin:
        utterances = json.load(fin)

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

            if "upperbound" in args.tag:
                knowledge_lki = uttdict["label"]
                utterance = uttdict["text"]
            content = utterance

            # Forward first pass
            prompt = templates[LLMtype]["slot"][1].format(**locals())
            outputs_list = []
            for tokenizer, model in zip(tokenizer_list, model_list):
                tokenized_seq = tokenizer(prompt, return_tensors="pt").input_ids.to(model.llm.device)
                outputs = model.generate_beam(
                    input_ids=tokenized_seq,
                    max_new_tokens=60,
                    beamsize=5,
                    n_adapters=1
                )
                lengths = torch.tensor([len(hyp.yseq) for hyp in outputs]).to(model.llm.device)
                logplist = torch.stack([hyp.cumscore for hyp in outputs])
                predictive_entropy, unnorm_entropy, _ = calc_predictive_entropy(logplist, 1, lengths)
                for k, hyp in enumerate(outputs):
                    output_txt = tokenizer.decode(hyp.yseq, skip_special_tokens=True).strip().split("</s>")[0]
                    empty = True
                    for char in output_txt:
                        if char not in string.punctuation:
                            empty = False
                    if empty:
                        output_txt = "{}"
                    outputs_list.append([output_txt, predictive_entropy])
            
            filtered_list = []
            seen = set()
            for result in outputs_list:
                if result[0] not in seen:
                    filtered_list.append(result)
                    seen.add(result[0])

            for i, result in enumerate(filtered_list):
                scores = torch.stack([model.scoring(prompt, result[0]) for model in model_list])
                filtered_list[i].append(scores)
                weight = torch.Tensor([0.5, 0.2, 0.3]).to(scores.device)
                scores = scores.matmul(weight)
                filtered_list[i].append(scores)
            best_output = max(filtered_list, key=lambda x: x[3].item())
            
            output = merge_outputs(best_output[0], slotdict)
            # print(predictive_entropy, unnorm_entropy, output)
            print(best_output)
            entity_f1, slu_f1 = calc_metrics(output, uttdict["label"])

            outputdict[slurpid] = {
                "output": output,
                "predictive": best_output[1].item(),
                # "unnormalised": unnorm_entropy.item(),
                "entity_f1": entity_f1,
                "slu_f1": slu_f1,
                "beamsearch_entropy": 0,
                "seg_correctness": 0,
                "seg_uncertainty": 0,
            }
            count += 1
            logging("Finished {}, Elapsed time {:.2f}".format(count, time.time()-start_time))

    with open(os.path.join(args.model_path, args.result_file), "w") as fout:
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
    ),
    parser.add_argument(
        "--result_file",
        type=str,
        default="output_weak_medium_top1upperbound_iter1.json",
        help="Inference result file name"
    )
    args = parser.parse_args()
    main(args)