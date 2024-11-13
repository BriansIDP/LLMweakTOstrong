# import debugpy

# # 5678是debugpy服务器监听的端口号，确保这个端口在你的系统上是空闲的
# debugpy.listen(('0.0.0.0', 5679))
# print("⏳ Waiting for debugger to attach...")

# # 让debugpy等待VSCode的调试器连接
# debugpy.wait_for_client()
# print("🚀 Debugger attached!")


import os
import random
import argparse
import math
import time
import copy
import json
import string
from collections import OrderedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import SchedulerType, AdamW, get_scheduler
from transformers.modeling_utils import load_sharded_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader

from dataset_dpo import (collate_fn, ActiveDataset, collate_fn_active, collate_fn_multiweak, wp_word_map, 
                     collate_fn_strongscore_mw, collate_fn_strongscore_mw_dpo, collate_fn_dpo, collate_fn_kto)
from data.prompt import prompts
from knowledgemodel import KnowledgeLLM
from scoring.evaluation.metrics import ErrorMetric
from scoring.evaluation.util import format_results, load_predictions, load_gold_data
from scoring.evaluation.normalizers.english import EnglishTextNormalizer
from loss import (logconf_loss_fn, logconf_step_loss_fn, logconf_confer_loss_fn, DirLossFn, 
                  edl_log_loss_fn, edl_logconf_loss_fn, edl_logconf_step_loss_fn, edl_logconf_confer_loss_fn,
                  soft_kl_loss_fn, soft_step_conf_loss_fn, soft_confer_loss_fn)

normaliser = EnglishTextNormalizer()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def check_nan(tensor, name, data, step):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name} at step {step}")
        torch.save(data, os.path.join(args.outputdir, "nan_data.pt"))
        raise ValueError(f"NaN detected in {name} at step {step}")


def get_grouped_params(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {   
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, # args.weight_decay,
        },
        {   
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def main(args):
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Save model configuration
    with open(os.path.join(args.outputdir, 'model_config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(args.lora_config) as fin:
        peft_params = json.load(fin)
    os.system("cp {} {}".format(args.lora_config, os.path.join(args.outputdir, 'lora_config.json')))
    os.system("cp {} {}".format("train_weak_to_strong_dpo.py", os.path.join(args.outputdir, "train.py")))
    os.system("cp {} {}".format("knowledgemodel.py", os.path.join(args.outputdir, "model.py")))
    os.system("cp {} {}".format("loss.py", os.path.join(args.outputdir, "loss.py")))

    ## Meta data
    with open("data/slotlist{}.json".format("_zero" if "_zero" in args.outputdir else "")) as fin:
        slotdict = json.load(fin)
        slotstr = ", ".join(['"' + key + '"' for key in slotdict.keys()])
    with open(args.ontology) as fin:
        knowledgebase = json.load(fin)

    ## Initialise data
    LLMtype = "vicuna"
    # if "llama-2" in args.model_path:
    #     LLMtype = "llama2"

    ##########################################
    # Load weak model first
    ##########################################
    if args.task == "normal":
        weak_tokenizer = AutoTokenizer.from_pretrained(args.weak_model_path, use_fast=("pythia" in args.weak_model_path or "bloom" in args.weak_model_path), trust_remote_code=True)
        if os.path.exists(args.pretrained_weak_model_path):
            if os.path.exists(os.path.join(args.pretrained_weak_model_path, "model.safetensors")):
                weakllm = AutoModelForCausalLM.from_pretrained(
                    os.path.join(args.pretrained_weak_model_path),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                    device_map="auto",
                )
            else:
                weakllm = AutoModelForCausalLM.from_pretrained(
                    args.weak_model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                    device_map="auto",
                )
                if os.path.exists(os.path.join(args.pretrained_weak_model_path, "pytorch_model.pt")):
                    state_dict = torch.load(os.path.join(args.pretrained_weak_model_path, "pytorch_model.pt"))
                    weakllm.load_state_dict(state_dict)
                else:
                    load_sharded_checkpoint(weakllm, os.path.join(args.pretrained_weak_model_path))
            weakmodel = KnowledgeLLM(weakllm, weak_tokenizer)
            weakmodel.eval()
        else:
            print("Error: Please input correct pretrained weak model path.")
            return 0
        
    elif args.task == "human_annotation":
        weak_tokenizer = AutoTokenizer.from_pretrained(args.weak_model_path, 
                                                       use_fast=("pythia" in args.weak_model_path or "bloom" in args.weak_model_path), 
                                                       trust_remote_code=True)
        
    elif args.task == "multi_weak" or args.task == "joint_decode":
        weak_model_list = []
        weak_tokenizer_list = []
        weak_model_names = args.weak_model_names.split(",")
        for weak_model_name in weak_model_names:
            pretrained_weak_model_path = os.path.join("exp/weak", weak_model_name, 'checkpoint.best')
            weak_model_path = os.path.join("/mnt/nvme_share/cuizy/models", weak_model_name)
            weak_tokenizer = AutoTokenizer.from_pretrained(weak_model_path, use_fast=("pythia" in args.weak_model_path or "bloom" in args.weak_model_path), trust_remote_code=True)
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
                del weakllm
                weakmodel.eval()
            else:
                print("Error: Please input correct pretrained weak model path.")
                return 0
            weak_model_list.append(weakmodel)
            weak_tokenizer_list.append(weak_tokenizer)
    
    
    ##########################################
    # Train MAIN model
    ##########################################
    ## Initialise data
    traindata = ActiveDataset(
        args.train_data_path,
        weak_tokenizer,
        slotdict,
        slotstr,
        prompts,
        asrplace=args.asrplace,
        num_candidates=args.num_candidates,
    )
    valdata = ActiveDataset(
        args.val_data_path,
        weak_tokenizer,
        slotdict,
        slotstr,
        prompts,
        asrplace=args.asrplace,
        num_candidates=args.num_candidates,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, 
                                              trust_remote_code=True, 
                                              use_fast=("pythia" in args.weak_model_path or "bloom" in args.weak_model_path),
                                              )
    # Initialise model
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        # torch_dtype=torch.float32,
        device_map="auto",
    )
    pretrained = False
    if os.path.exists(args.pretrained_model_path):
        load_sharded_checkpoint(llm, args.pretrained_model_path)
        pretrained = True
    # model = KnowledgeLLM(llm, tokenizer).to(device)
    model = KnowledgeLLM(llm, tokenizer)
    # del llm

    if args.kto:
        ref_llm = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map="auto",
        )
        load_sharded_checkpoint(ref_llm, args.pretrained_model_path)
        ref_model = KnowledgeLLM(ref_llm, tokenizer)
        # del ref_llm
        ref_model.eval()
    
    if not pretrained:
        with torch.no_grad():
            traindata.refill_labelset(step=0)
            if args.task == "normal":
                traindata = get_next_labelset(args, weak_tokenizer, weakmodel, traindata)
                weakmodel.cpu()
            elif args.task == "multi_weak":
                traindata = get_next_labelset_multiweak(args, weak_tokenizer_list, weak_model_list, traindata)
                del weakmodel
            elif args.task == "joint_decode":
                traindata = get_next_labelset_jointdecode(args, weak_tokenizer_list, weak_model_list, traindata)
            traindata.tokenizer = tokenizer
            valdata.refill_labelset(step=0)
            valdata.tokenizer = tokenizer
        if args.task == "multi_weak":
            if args.strong_score_wordpiece:
                traindata.strong_score = True
                train_dataloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_strongscore_mw)
            else:
                train_dataloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_multiweak)
        else:
            train_dataloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valdata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        num_train_epochs = args.num_train_epochs
    else:
        traindata.refill_labelset(step=0)
        traindata.tokenizer = tokenizer
        valdata.refill_labelset(step=0)
        valdata.tokenizer = tokenizer
        train_dataloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valdata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        num_train_epochs = 0

    # Initialise criterion
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    if args.criterion == "xent":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif args.criterion == "logconf":
        criterion = logconf_loss_fn()
    elif args.criterion == "logconf_step":
        criterion = logconf_step_loss_fn()
    elif args.criterion == "logconf_confer":
        criterion = logconf_confer_loss_fn()
    elif args.criterion == "dir":
        criterion = DirLossFn()
    elif args.criterion == "edl":
        criterion = edl_log_loss_fn()
    elif args.criterion == "edl_conf":
        criterion = edl_logconf_loss_fn()
    elif args.criterion == "edl_step":
        criterion = edl_logconf_step_loss_fn()
    elif args.criterion == "edl_confer":
        criterion = edl_logconf_confer_loss_fn()
    elif args.criterion == "soft":
        criterion = soft_kl_loss_fn()
    elif args.criterion == "soft_step":
        criterion = soft_step_conf_loss_fn()
    elif args.criterion == "soft_confer":
        criterion = soft_confer_loss_fn()

    optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
    num_update_steps_per_epoch = math.ceil(len(traindata) / (args.gradient_accumulation_steps * args.batch_size))
    max_train_steps = (num_train_epochs + args.num_train_epochs_dpo) * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    print("Start training MAIN MODEL")
    best_val_loss = 10000
    for epoch in range(num_train_epochs):
        model.train()
        if args.task == "multi_weak" and args.strong_score_wordpiece:
            model = train_one_epoch_multiweak(
                args,
                epoch,
                model,
                train_dataloader,
                optimizer,
                lr_scheduler,
                criterion=criterion,
                tokenizer=tokenizer,
            )
        else:
            model = train_one_epoch(
                args,
                epoch,
                model,
                train_dataloader,
                optimizer,
                lr_scheduler,
                criterion=criterion,
                tokenizer=tokenizer,
            )
        model.eval()
        current_lr = optimizer.param_groups[0]["lr"]
        save_checkpoint(model, tokenizer, args.outputdir, epoch)
    
    print("Start update dataset for dpo")
    with torch.no_grad():
        traindata = get_next_labelset_dpo(args, tokenizer, model, weak_model_list, traindata)
    if not args.kto:
        train_dataloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_dpo)
        del weakmodel
        del weak_model_list
        torch.cuda.empty_cache()
        for epoch in range(args.num_train_epochs_dpo):
            model.train()
            model = train_one_epoch_dpo(
                args, 
                epoch+num_train_epochs, 
                model, 
                train_dataloader, 
                optimizer, 
                lr_scheduler,
                criterion=criterion,
                tokenizer=tokenizer,
                )
            save_checkpoint(model, tokenizer, args.outputdir, epoch+num_train_epochs)
    else:
        traindata.dpo = False
        traindata.kto = True
        train_dataloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_kto)
        del weakmodel
        del weak_model_list
        for epoch in range(args.num_train_epochs_dpo):
            model.train()
            model = train_one_epoch_kto(
                args, 
                epoch+num_train_epochs, 
                model, 
                train_dataloader, 
                optimizer, 
                lr_scheduler,
                criterion=criterion,
                ref_model=ref_model,
                tokenizer=tokenizer,
                )
            save_checkpoint(model, tokenizer, args.outputdir, epoch+num_train_epochs)


def save_checkpoint(model, tokenizer, outputdir, epoch):
    fulloutput = os.path.join(outputdir, "checkpoint.{}".format(epoch))
    os.system(f"mkdir -p {fulloutput}")
    checkpoint = OrderedDict()
    # save tokenizer
    tokenizer.save_pretrained(fulloutput)
    # save configuration
    # if "gpt2" in model.llm.config._name_or_path:
    #     torch.save(model.llm.state_dict(), os.path.join(fulloutput, "pytorch_model.pt"))
    # else:
    model.llm.save_pretrained(fulloutput)
    return checkpoint


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
        try:
            new_output[key] = " & ".join(value)
        except:
            continue
    return json.dumps(new_output)


def get_cascaded_uncertainty(model, prompt_nbest, generate_hyps, tokenizer, lengths, device):
    T = 0.1 # 0.001
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
    forward_logp = forward_logp.view(-1, forward_logp.size(-1))
    forward_logp = forward_logp[torch.arange(forward_logp.size(0)), labels.reshape(-1)].reshape(labels.size(0), -1)

    seq_logp = (forward_logp * label_mask).sum(dim=-1)
    seq_logp = seq_logp.view(nasrhyps, nhyps)
    cascaded_entropies = []
    entropy, unnorm_entropy, _ = calc_predictive_entropy(seq_logp, 1.0, lengths.unsqueeze(0))
    entropy = (entropy * asr_dist).sum(dim=0)
    return entropy


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


def get_next_labelset(args, tokenizer, model, traindata, strongerset=None):
    traindata.preprocess = False
    if strongerset is not None:
        stronger_tokenizer, strongermodel = strongerset
    active_loader = DataLoader(traindata, batch_size=1, shuffle=False, collate_fn=collate_fn_active)
    firstpass_ids_dict = {}
    uncertainties = []
    count = 0
    for batch in tqdm(active_loader):
        slurp_ids, sequences, nbest, label = batch
        tokenized_seq = tokenizer(sequences[0], return_tensors="pt").input_ids.to(model.llm.device)
        outputs = model.generate_beam(
            tokenized_seq,
            max_new_tokens=64,
            beamsize=5,
        )
        firstpass_ids_dict[slurp_ids[0]] = []
        lengths = torch.tensor([len(hyp.yseq) for hyp in outputs]).to(model.llm.device)
        logplist = torch.stack([hyp.cumscore for hyp in outputs])
        predictive_entropy, unnorm_entropy, _ = calc_predictive_entropy(logplist, 1, lengths)
        uncertainties.append(predictive_entropy)
        for k, hyp in enumerate(outputs):
            # firstpass_ids_dict[slurp_ids[0]].append([tokenizer.decode(hyp.yseq).split("</s>")[0], predictive_entropy])
            output_txt = tokenizer.decode(hyp.yseq, skip_special_tokens=True).strip().split("</s>")[0]
            wp_score = torch.Tensor(hyp.scores)
            # if output_txt == "":
            #     output_txt = "{}"
            empty = True
            for char in output_txt:
                if char not in string.punctuation:
                    empty = False
            if empty:
                output_txt = "{}"
                word_score = torch.sum(wp_score)
            else:
                weak_input_ids = tokenizer.encode(output_txt+"</s>")
                mapping = wp_word_map(
                    wordpiece_list=[tokenizer.decode(id).replace(' ', '') for id in weak_input_ids],
                    word_list=(output_txt+"</s>").split(' ')
                )
                word_score = torch.stack([wp_score[start:end+1].sum() for start, end in mapping])
            firstpass_ids_dict[slurp_ids[0]].append([output_txt, predictive_entropy, word_score])
            # firstpass_ids_dict[slurp_ids[0]].append([tokenizer.decode(hyp.yseq, skip_special_tokens=True), predictive_entropy])
    uncertainties = sorted(uncertainties, reverse=True)
    threshold = uncertainties[int(args.unc_threshold * len(uncertainties))]
    logging(f"Threshold for uncertainty: {threshold}")
    traindata.update_with_firstpass(
        firstpass_ids_dict,
        threshold,
        # update_label=strongerset is not None,
        update_label=True,
        update_delib=args.task=="deliberation",
    )
    traindata.preprocess = True
    return traindata


def get_next_labelset_multiweak(args, weak_tokenizer_list, weak_model_list, traindata):
    traindata.preprocess = False
    active_loader = DataLoader(traindata, batch_size=1, shuffle=False, collate_fn=collate_fn_active)
    firstpass_ids_dict = {}
    uncertainties = []
    for batch in tqdm(active_loader):
        slurp_ids, sequences, nbest, label = batch
        outputs_list = []
        for tokenizer, model in zip(weak_tokenizer_list, weak_model_list):
            tokenized_seq = tokenizer(sequences[0], return_tensors="pt").input_ids.to(model.llm.device)
            outputs = model.generate_beam(
                tokenized_seq,
                max_new_tokens=64,
                beamsize=5,
            )
            lengths = torch.tensor([len(hyp.yseq) for hyp in outputs]).to(model.llm.device)
            logplist = torch.stack([hyp.cumscore for hyp in outputs])
            predictive_entropy, unnorm_entropy, _ = calc_predictive_entropy(logplist, 1, lengths)
            # outputs_list.append([tokenizer.decode(outputs[0].yseq).split("</s>")[0], predictive_entropy])
            # outputs_list.append([tokenizer.decode(outputs[0].yseq, skip_special_tokens=True).strip(), predictive_entropy])
            output_txt = tokenizer.decode(outputs[0].yseq, skip_special_tokens=True).strip().split("</s>")[0]
            wp_score = torch.Tensor(outputs[0].scores)
            empty = True
            for char in output_txt:
                if char not in string.punctuation:
                    empty = False
            if empty:
                output_txt = "{}"
                word_score = torch.sum(wp_score)
            else:
                weak_input_ids = tokenizer.encode(output_txt+"</s>")
                mapping = wp_word_map(
                    wordpiece_list=[tokenizer.decode(id).replace(' ', '') for id in weak_input_ids],
                    word_list=(output_txt+"</s>").split(' ')
                )
                word_score = torch.stack([wp_score[start:end+1].sum() for start, end in mapping])

            outputs_list.append([output_txt, predictive_entropy, word_score])
            uncertainties.append(predictive_entropy)
        firstpass_ids_dict[slurp_ids[0]] = outputs_list
    uncertainties = sorted(uncertainties, reverse=True)
    threshold = uncertainties[int(args.unc_threshold * len(uncertainties))]
    logging(f"Threshold for uncertainty: {threshold}")
    traindata.update_with_firstpass_multiweak(
        labelset=firstpass_ids_dict,
        threshold=threshold,
        update_label=True
    )
    traindata.preprocess = True
    traindata.multi_weak = True
    return traindata


def get_next_labelset_jointdecode(args, weak_tokenizer_list, weak_model_list, traindata):
    traindata.preprocess = False
    active_loader = DataLoader(traindata, batch_size=1, shuffle=False, collate_fn=collate_fn_active)
    firstpass_ids_dict = {}
    uncertainties = []
    for batch in tqdm(active_loader):
        slurp_ids, sequences, nbest, label = batch
        outputs_list = []
        value_list = []
        for tokenizer, model in zip(weak_tokenizer_list, weak_model_list):
            tokenized_seq = tokenizer(sequences[0], return_tensors="pt").input_ids.to(model.llm.device)
            outputs = model.generate_beam(
                tokenized_seq,
                max_new_tokens=64,
                beamsize=5,
            )
            lengths = torch.tensor([len(hyp.yseq) for hyp in outputs]).to(model.llm.device)
            logplist = torch.stack([hyp.cumscore for hyp in outputs])
            predictive_entropy, unnorm_entropy, _ = calc_predictive_entropy(logplist, 1, lengths)
            for k, hyp in enumerate(outputs):
                output_txt = tokenizer.decode(hyp.yseq, skip_special_tokens=True).strip().split("</s>")[0]
                wp_score = torch.Tensor(hyp.scores)
                # if output_txt == "":
                #     output_txt = "{}"
                empty = True
                for char in output_txt:
                    if char not in string.punctuation:
                        empty = False
                if empty:
                    output_txt = "{}"
                    word_score = torch.sum(wp_score)
                else:
                    weak_input_ids = tokenizer.encode(output_txt+"</s>")
                    mapping = wp_word_map(
                        wordpiece_list=[tokenizer.decode(id).replace(' ', '') for id in weak_input_ids],
                        word_list=(output_txt+"</s>").split(' ')
                    )
                    word_score = torch.stack([wp_score[start:end+1].sum() for start, end in mapping])
                outputs_list.append([output_txt, predictive_entropy, word_score])
            value_list.append(predictive_entropy)

        filtered_list = []
        seen = set()
        for result in outputs_list:
            if result[0] not in seen:
                filtered_list.append(result)
                seen.add(result[0])

        value = torch.stack(value_list).float()
        value = torch.softmax(1/torch.sqrt(value), dim=-1)
        for i, result in enumerate(filtered_list):
            scores = torch.stack([model.scoring(sequences[0], result[0]) for model in weak_model_list])
            # filtered_list[i].append(scores.mean())
            filtered_list[i].append(scores)
            weight = torch.Tensor([0.5, 0.2, 0.3]).to(scores.device)
            scores = scores.matmul(weight)
            # scores = scores.matmul(value)
            # scores = scores.mean()
            filtered_list[i].append(scores)
        best_output = max(filtered_list, key=lambda x: x[4].item())

        firstpass_ids_dict[slurp_ids[0]] = [best_output]
        uncertainties.append(best_output[1])
    uncertainties = sorted(uncertainties, reverse=True)
    threshold = uncertainties[int(args.unc_threshold * len(uncertainties))]
    logging(f"Threshold for uncertainty: {threshold}")
    traindata.update_with_firstpass(
        labelset=firstpass_ids_dict,
        threshold=threshold,
        update_label=True
    )
    traindata.preprocess = True
    return traindata


def get_next_labelset_dpo(args, strong_tokenizer, strong_model, weak_model_list, traindata):
    traindata.preprocess = False
    traindata.multi_weak = False
    active_loader = DataLoader(traindata, batch_size=1, shuffle=False, collate_fn=collate_fn_active)
    firstpass_ids_dict = {}
    uncertainties = []
    for batch in tqdm(active_loader):
        slurp_ids, sequences, nbest, label = batch
        tokenized_seq = strong_tokenizer(sequences[0], return_tensors="pt").input_ids.to(strong_model.llm.device)
        outputs = strong_model.generate_beam(
            tokenized_seq,
            max_new_tokens=64,
            beamsize=10,
        )
        lengths = torch.tensor([len(hyp.yseq) for hyp in outputs]).to(strong_model.llm.device)
        logplist = torch.stack([hyp.cumscore for hyp in outputs])
        predictive_entropy, unnorm_entropy, _ = calc_predictive_entropy(logplist, 1, lengths)
        uncertainties.append(predictive_entropy)
        output_list = []
        for k, hyp in enumerate(outputs):
            output_txt = strong_tokenizer.decode(hyp.yseq)
            # output_txt = strong_tokenizer.decode(hyp.yseq, skip_special_tokens=True).strip().split("</s>")[0]
            wp_score = torch.Tensor(hyp.scores)
            label_ids = torch.IntTensor(hyp.yseq)
            assert len(label_ids) == len(wp_score)
            scores = torch.stack([model.scoring(sequences[0], output_txt) for model in weak_model_list])
            if len(weak_model_list) == 3:
                weight = torch.Tensor([0.3, 0.3, 0.4]).to(scores.device)
            elif len(weak_model_list) == 5:
                weight = torch.Tensor([0.1, 0.3, 0.2, 0.3, 0.1]).to(scores.device)
            scores = scores.matmul(weight)
            # output_list.append([output_txt, wp_score, scores])
            output_list.append([label_ids, wp_score, scores])
        chosen_output = max(output_list, key=lambda x: x[2].item())
        rejected_output = min(output_list, key=lambda x: x[2].item())
        firstpass_ids_dict[slurp_ids[0]] = [chosen_output[0], predictive_entropy, chosen_output[1],
                                            rejected_output[0], rejected_output[1]]
    traindata.update_with_firstpass_dpo(
        labelset=firstpass_ids_dict,
    )
    traindata.preprocess = True
    traindata.dpo = True
    traindata.multiweak = False
    return traindata


def calc_predictive_entropy(logp, temperature, lengths):
    pi_b = torch.softmax(logp / temperature, dim=-1)
    entropy = - (logp * pi_b / lengths).sum(dim=-1)
    entropy_seq = - logp * pi_b / lengths
    entropy_unnorm = - (logp * pi_b).sum(dim=-1)
    return entropy, entropy_unnorm, entropy_seq


def train_one_epoch(args, epoch, model, train_dataloader, optimizer, lr_scheduler, criterion, knowledge=None, tokenizer=None):
    optimizer.zero_grad()
    trainsize = len(train_dataloader)
    start = time.time()
    kgloss = 0
    for i, batch in enumerate(train_dataloader):
        inputs, labels, nbest_prompt, values, token_scores = batch
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output, labels = model(
                inputs,
                labels,
                knowledge=knowledge,
            )
            logits = output.logits
            seplosses = None
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(logits.view(-1, logits.size(-1)), labels.reshape(-1))
            elif "soft" in args.criterion or "dir" in args.criterion or "edl" in args.criterion:
                step_frac = (len(train_dataloader) * epoch + i) / len(train_dataloader) / args.num_train_epochs
                loss = criterion(logits, labels, step_frac, values, token_scores)
            else:
                step_frac = (len(train_dataloader) * epoch + i) / len(train_dataloader) / args.num_train_epochs
                loss = criterion(logits, labels, step_frac, values)
            loss = loss / args.gradient_accumulation_steps
        check_nan(loss, "loss", (inputs, labels, logits, loss, token_scores), i+1)
        loss.backward()

        for name, param in model.llm.named_parameters():
            if param.grad is not None:
                check_nan(param.grad, f"gradient of {name}", (inputs, labels, logits, loss, token_scores), i+1)

        if (i + 1) % args.gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if (i + 1) % args.log_interval == 0:
            elasped_time = time.time() - start
            PPL = math.exp(loss.item() * args.gradient_accumulation_steps)
            logging(f"Epoch {epoch} | Batch {i}/{trainsize} | PPL: {PPL} | time {elasped_time}")
    return model


def train_one_epoch_dpo(args, epoch, model, train_dataloader, optimizer, lr_scheduler, criterion, knowledge=None, tokenizer=None):
    optimizer.zero_grad()
    trainsize = len(train_dataloader)
    start = time.time()
    kgloss = 0
    for i, batch in enumerate(train_dataloader):
        chosen_inputs, chosen_labels, chosen_scores, rejected_inputs, rejected_labels, rejected_scores = batch
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            chosen_output, chosen_labels = model(
                chosen_inputs,
                chosen_labels,
                knowledge=knowledge,
            )
            chosen_logits = chosen_output.logits
            # seplosses = None
            # if isinstance(criterion, torch.nn.CrossEntropyLoss):
            #     chosen_loss = criterion(chosen_logits.view(-1, chosen_logits.size(-1)), chosen_labels.reshape(-1))
            # elif "soft" in args.criterion or "dir" in args.criterion or "edl" in args.criterion:
            #     step_frac = (len(train_dataloader) * epoch + i) / len(train_dataloader) / args.num_train_epochs
            #     chosen_loss = criterion(chosen_logits, chosen_labels, step_frac, torch.Tensor([0]), chosen_scores)
            # else:
            #     step_frac = (len(train_dataloader) * epoch + i) / len(train_dataloader) / args.num_train_epochs
            #     chosen_loss = criterion(chosen_logits, chosen_labels, step_frac, torch.Tensor([0]))
            # chosen_loss = chosen_loss / args.gradient_accumulation_steps
            chosen_logps = get_logprobs(chosen_logits, chosen_labels, average_log_prob=False)
            chosen_ref_logps = chosen_scores.sum()

            rejected_output, rejected_labels = model(
                rejected_inputs,
                rejected_labels,
                knowledge=knowledge,
            )
            rejected_logits = rejected_output.logits
            # seplosses = None
            # if isinstance(criterion, torch.nn.CrossEntropyLoss):
            #     rejected_loss = criterion(rejected_logits.view(-1, rejected_logits.size(-1)), rejected_labels.reshape(-1))
            # elif "soft" in args.criterion or "dir" in args.criterion or "edl" in args.criterion:
            #     step_frac = (len(train_dataloader) * epoch + i) / len(train_dataloader) / args.num_train_epochs
            #     rejected_loss = criterion(rejected_logits, rejected_labels, step_frac, torch.Tensor([0]), rejected_scores)
            # else:
            #     step_frac = (len(train_dataloader) * epoch + i) / len(train_dataloader) / args.num_train_epochs
            #     rejected_loss = criterion(rejected_logits, rejected_labels, step_frac, torch.Tensor([0]))
            # rejected_loss = rejected_loss / args.gradient_accumulation_steps
            rejected_logps = get_logprobs(rejected_logits, rejected_labels, average_log_prob=False)
            rejected_ref_logps = rejected_scores.sum()

            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = chosen_ref_logps - rejected_ref_logps
            logits = pi_logratios - ref_logratios
            loss = - torch.nn.functional.logsigmoid(args.beta * logits) * (1 - args.label_smoothing) - torch.nn.functional.logsigmoid(-args.beta * logits) * args.label_smoothing
            # loss = (logits - 1/(2 * args.beta)) ** 2
        
        # loss = chosen_loss - rejected_loss * 0.2
        # loss = -torch.nn.functional.logsigmoid(0.1 * (chosen_loss - rejected_loss))
        # check_nan(loss, "loss", (chosen_inputs, labels, logits, loss, token_scores), i+1)
        (loss / args.gradient_accumulation_steps).backward()

        # for name, param in model.llm.named_parameters():
        #     if param.grad is not None:
        #         check_nan(param.grad, f"gradient of {name}", (inputs, labels, logits, loss, token_scores), i+1)

        if (i + 1) % args.gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if (i + 1) % args.log_interval == 0:
            elasped_time = time.time() - start
            PPL = math.exp(loss.item())
            logging(f"Epoch {epoch} | Batch {i}/{trainsize} | PPL: {PPL} | time {elasped_time}")
    return model


def train_one_epoch_kto(args, epoch, model, train_dataloader, optimizer, lr_scheduler, criterion, ref_model, knowledge=None, tokenizer=None):
    optimizer.zero_grad()
    trainsize = len(train_dataloader)
    start = time.time()
    for i, batch in enumerate(train_dataloader):
        chosen_inputs, chosen_labels, chosen_scores, rejected_inputs, rejected_labels, rejected_scores, kl_inputs, kl_labels = batch
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            chosen_output, chosen_labels = model(
                chosen_inputs,
                chosen_labels,
                knowledge=knowledge,
            )
            chosen_logits = chosen_output.logits
            chosen_logps = get_logprobs(chosen_logits, chosen_labels, average_log_prob=False)
            chosen_ref_logps = chosen_scores.sum()

            rejected_output, rejected_labels = model(
                rejected_inputs,
                rejected_labels,
                knowledge=knowledge,
            )
            rejected_logits = rejected_output.logits
            rejected_logps = get_logprobs(rejected_logits, rejected_labels, average_log_prob=False)
            rejected_ref_logps = rejected_scores.sum()

            kl_output, kl_labels = model(
                kl_inputs,
                kl_labels,
                knowledge=knowledge,
            )
            kl_logits = kl_output.logits
            kl_logps = get_logprobs(kl_logits, kl_labels, average_log_prob=False)
            with torch.no_grad():
                kl_ref_output, kl_ref_labels = ref_model(
                    kl_inputs,
                    kl_labels,
                    knowledge=knowledge,
                )
                kl_ref_logits = kl_ref_output.logits
                kl_ref_logps = get_logprobs(kl_ref_logits, kl_ref_labels, average_log_prob=False)
            KL = (kl_logps - kl_ref_logps).detach()

            chosen_logratios = (chosen_logps - chosen_ref_logps)
            chosen_loss = 1 - torch.nn.functional.sigmoid(args.beta * (chosen_logratios - KL))
            rejected_logratios = (rejected_logps - rejected_ref_logps)
            rejected_loss = 1 - torch.nn.functional.sigmoid(args.beta * (KL - rejected_logratios))
            
            # loss = torch.stack([chosen_loss, rejected_loss]).mean()
            loss = (chosen_loss + rejected_loss) / 2

        (loss / args.gradient_accumulation_steps).backward()

        if (i + 1) % args.gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if (i + 1) % args.log_interval == 0:
            elasped_time = time.time() - start
            PPL = math.exp(loss.item())
            logging(f"Epoch {epoch} | Batch {i}/{trainsize} | PPL: {PPL} | time {elasped_time}")
    return model


def train_one_epoch_multiweak(args, epoch, model, train_dataloader, optimizer, lr_scheduler, criterion, knowledge=None, tokenizer=None):
    optimizer.zero_grad()
    trainsize = len(train_dataloader)
    start_time = time.time()
    kgloss = 0
    for i, batch in enumerate(train_dataloader):
        inputs_list, total_label_list, nbest_propmpt, values_list, score_list, mapping_list = batch
        # loss = 0
        loss = []
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for inputs, labels, values, wd_score, mapping in zip(inputs_list, total_label_list, values_list, score_list, mapping_list):
                for key in inputs:
                    inputs[key] = inputs[key].to(model.llm.device)
                labels = labels.to(model.llm.device)
                values = values.to(model.llm.device)
                wd_score = wd_score.to(model.llm.device)
                if wd_score.numel() == 1:
                    wd_score = wd_score.unsqueeze(0)

                output, labels = model(inputs, labels, knowledge=knowledge)
                logits = output.logits

                labels = labels.reshape(-1)
                lossmask = (labels == -1)
                logits = logits.reshape(-1, logits.size(-1))
                logits = logits[lossmask == False]
                labels = labels[lossmask == False]
                probs = torch.softmax(logits, dim=-1)
                wp_score_list = []
                for j, (start, end) in enumerate(mapping):
                    strong_probs, _ = torch.max(probs[start:end+1], dim=-1)
                    wp_score = torch.softmax(-strong_probs/0.3, dim=0) * wd_score[j]
                    wp_score_list.append(wp_score.detach())
                score = torch.concat(wp_score_list, dim=0)
                assert len(score) == len(labels)

                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    loss.append(criterion(logits.view(-1, logits.size(-1)), labels.reshape(-1)))
                elif "soft" in args.criterion or "dir" in args.criterion or "edl" in args.criterion:
                    step_frac = (len(train_dataloader) * epoch + i) / len(train_dataloader) / args.num_train_epochs
                    loss.append(criterion(logits, labels, step_frac, values, score))
                else:
                    step_frac = (len(train_dataloader) * epoch + i) / len(train_dataloader) / args.num_train_epochs
                    loss.append(criterion(logits, labels, step_frac, values))
            # loss = loss / args.gradient_accumulation_steps / len(inputs_list)
            loss = torch.stack(loss)
            loss = loss.mean()
            weight = torch.Tensor([0.25, 0.25, 0.5]).to(loss.device)
            # loss = loss.matmul(weight)
            loss = loss / args.gradient_accumulation_steps
            
        loss.backward()

        if (i + 1) % args.gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if (i + 1) % args.log_interval == 0:
            elasped_time = time.time() - start_time
            PPL = math.exp(loss.item() * args.gradient_accumulation_steps)
            logging(f"Epoch {epoch} | Batch {i}/{trainsize} | PPL: {PPL} | time {elasped_time}")

    return model


def eval_one_epoch(args, epoch, model, valid_dataloader, criterion, knowledge=None, tokenizer=None):
    total_tokens = 0
    total_loss = 0.
    total_kgloss = 0.
    total_kgtokens = 0
    for i, batch in enumerate(valid_dataloader):
        inputs, labels, nbest_prompt, values, scores = batch
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output, labels = model(
                inputs,
                labels,
                knowledge=knowledge,
            )
            logits = output.logits
            # loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(logits.view(-1, logits.size(-1)), labels.reshape(-1))
            else:
                step_frac = epoch / args.num_train_epochs
                loss = criterion(logits, labels, step_frac, values)
        tokens = (labels != -1).sum()
        total_tokens += tokens
        total_loss += loss.item() * tokens
    val_loss = total_loss / total_tokens
    return val_loss


def get_logprobs(logits, labels, average_log_prob=False):
    num_class = logits.size(-1)
    logits = logits.float().view(-1, num_class)
    labels = labels.reshape(-1)
    lossmask = (labels == -1)
    logits = logits[lossmask == False]
    labels = labels[lossmask == False]
    logprobs = torch.log_softmax(logits, dim=-1)
    label_onehot = torch.nn.functional.one_hot(labels, num_classes=num_class)
    logprobs = logprobs[label_onehot==1].view(-1, 1)
    if average_log_prob:
        return logprobs.mean()
    else:
        return logprobs.sum()


if __name__ == "__main__":
    ## Parameter groups
    parser = argparse.ArgumentParser(description="LLM finetuning")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./hf_models",
        help="Path to the model file",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="",
        help="Path to the pretrained model file",
    )
    parser.add_argument(
        "--weak_model_path",
        type=str,
        default="gpt2",
        help="Path to the weak supervisor model file",
    )
    parser.add_argument(
        "--strong_model_path",
        type=str,
        default="gpt2",
        help="Path to the weak supervisor model file",
    )
    parser.add_argument(
        "--pretrained_weak_model_path",
        type=str,
        default="",
        help="Path to the weak supervisor model file",
    )
    parser.add_argument(
        "--pretrained_strong_model_path",
        type=str,
        default="",
        help="Path to the weak supervisor model file",
    )
    parser.add_argument(
        "--weak_train_samples",
        type=int,
        default=2000,
        help="Data samples to train weak model",
    )
    parser.add_argument(
        "--strong_train_samples",
        type=int,
        default=2000,
        help="Data samples to train weak model",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./hf_models",
        help="Path to the train data file",
    )
    parser.add_argument(
        "--weak_train_path",
        type=str,
        default="./hf_models",
        help="Path to the weak model train data file",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="./hf_models",
        help="Path to the val data file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to the saved checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_epochs_dpo", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default='./exp/clip_vlm',
        help="Path to the output dir",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="log interval",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=1,
        help="Top n from the list to use",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default="",
        help="KB for biasing",
    )
    parser.add_argument(
        "--maxKBsize",
        type=int,
        default=10,
        help="Size of the biasing list to use",
    )
    parser.add_argument(
        "--KBdrop",
        type=float,
        default=0.0,
        help="Drop ratio for true biasing entities",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Schema config",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="xent",
        help="Loss function",
        choices=["xent", "logconf", "logconf_step", "logconf_confer", "dir", "edl", "edl_conf", "edl_step", "edl_confer",
                 "soft", "soft_conf", "soft_step", "soft_confer"]
    )
    parser.add_argument(
        "--use_lora",
        type=str,
        default="false",
        help="Use lora for finetuning",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="normal",
        choices=["normal", "deliberation", "human_annotation", "multi_weak", "joint_decode"],
        help="'normal' uses stronger model labels without deliberation. 'human_annotation' uses human labels. 'multi_weak' for multiple pretrained weak model.",
    )
    parser.add_argument(
        "--weak_model_names",
        type=str,
        default="gpt2-large,opt-1.3b,pythia-1.4b",
        help="name of weak models. Saved in exp/weak/{weak_model_name}, only applied when task == multi_weak." 
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="data/lora_config.json",
        help="Peft config file",
    )
    parser.add_argument(
        "--asrplace",
        type=str,
        default="none",
        help="Where to use ASR output, choose from none, weak, main, both",
    )
    parser.add_argument(
        "--unc_threshold",
        type=float,
        default=0.0,
        help="Portion to be considered as uncertain",
    )
    parser.add_argument(
        "--auxconfalpha",
        type=float,
        default=0.0,
        help="How much to use weak label",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="Size of ensemble",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--strong_score_wordpiece", 
        action="store_true",
        help="Whether to split wordpiece score according to strong model logits. "
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument(
        "--kto", 
        action="store_true",
        help="Use KTO if True else DPO."
    )
    args = parser.parse_args()
    main(args)
