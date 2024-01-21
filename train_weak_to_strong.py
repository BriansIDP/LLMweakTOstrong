import os
import random
import argparse
import math
import pickle
import time
import copy
import json
from collections import OrderedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import SchedulerType, AdamW, get_scheduler
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader

from dataset import SupervisedDataset, collate_fn, ActiveDataset, collate_fn_active
from data.prompt import prompts
from knowledgemodel import KnowledgeLLM


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
random.seed(1)
torch.manual_seed(1)

def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

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
    # Save model configuration
    with open(os.path.join(args.outputdir, 'model_config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(args.lora_config) as fin:
        peft_params = json.load(fin)
    os.system("cp {} {}".format(args.lora_config, os.path.join(args.outputdir, 'lora_config.json')))

    ## Meta data
    with open("data/slotlist{}.json".format("_zero" if "_zero" in args.outputdir else "")) as fin:
        slotdict = json.load(fin)
        slotstr = ", ".join(['"' + key + '"' for key in slotdict.keys()])
    with open(args.ontology) as fin:
        knowledgebase = json.load(fin)

    ## Initialise data
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    LLMtype = "vicuna"
    if "llama-2" in args.model_path:
        LLMtype = "llama2"
    weak_tokenizer = AutoTokenizer.from_pretrained(args.weak_model_path)

    ## Initialise data
    weaktraindata = ActiveDataset(
        args.train_data_path,
        weak_tokenizer,
        slotdict,
        slotstr,
        prompts,
    )
    weakvaldata = ActiveDataset(
        args.val_data_path,
        weak_tokenizer,
        slotdict,
        slotstr,
        prompts,
    )

    weaktraindata.refill_labelset(step=min(len(weaktraindata.data), args.weak_train_samples))
    weakvaldata.refill_labelset(step=0)
    train_dataloader = DataLoader(weaktraindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(weakvaldata, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    ##########################################
    # Train weak model first
    ##########################################
    # Define model
    weakllm = AutoModelForCausalLM.from_pretrained(
        args.weak_model_path,
        torch_dtype=torch.float16 if "gpt2" not in args.weak_model_path else torch.float32,
        cache_dir="/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMknowledge/cache",
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=peft_params["lora_rank"],
        lora_alpha=peft_params["lora_alpha"],
        lora_dropout=peft_params["lora_dropout"],
        target_modules=peft_params["lora_module"],
    )
    if "gpt2" not in args.weak_model_path:
        weakllm = get_peft_model(weakllm, peft_config)
        weakllm.print_trainable_parameters()
    weakmodel = KnowledgeLLM(weakllm, weak_tokenizer)
    weakmodel = weakmodel.to(device)

    ## Initialise criterion and optimiser
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    ## Optimiser
    optimizer = AdamW(get_grouped_params(weakmodel), lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(weaktraindata) / (args.gradient_accumulation_steps * args.batch_size))
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    print("Start training WEAK MODEL")
    best_val_loss = 10000
    best_weak_model = weakmodel
    for epoch in range(args.num_train_epochs):
        weakmodel.train()
        weakmodel = train_one_epoch(
            args,
            epoch,
            weakmodel,
            train_dataloader,
            optimizer,
            lr_scheduler,
            criterion,
            tokenizer=weak_tokenizer,
        )
        weakmodel.eval()
        with torch.no_grad():
            val_loss = eval_one_epoch(
                args,
                weakmodel,
                valid_dataloader,
                criterion,
                tokenizer=weak_tokenizer,
            )
        val_ppl = math.exp(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logging(f"WEAK MODEL Epoch {epoch} | Validation PPL: {val_ppl} | Learning rate: {current_lr}")
        # Save models
        if val_loss < best_val_loss:
            logging(f"Saving best WEAK MODEL at Epoch {epoch}")
            save_checkpoint(weakmodel, weak_tokenizer, args.outputdir, "best_weak")
            best_val_loss = val_loss
            best_weak_model = copy.deepcopy(weakmodel.state_dict())

    ##########################################
    # Train MAIN model
    ##########################################
    # Load best WEAK model
    weakmodel.load_state_dict(best_weak_model)

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
    with torch.no_grad():
        traindata.refill_labelset(step=0)
        traindata = get_next_labelset(args, weak_tokenizer, weakmodel, traindata)
        traindata.tokenizer = tokenizer
        valdata.refill_labelset(step=0)
        valdata = get_next_labelset(args, weak_tokenizer, weakmodel, valdata)
        valdata.tokenizer = tokenizer

    train_dataloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valdata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if "gpt2" not in args.model_path else torch.float32,
        cache_dir="/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMknowledge/cache",
    )
    if "gpt2" not in args.model_path:
        llm = get_peft_model(llm, peft_config)
        llm.print_trainable_parameters
    model = KnowledgeLLM(llm, tokenizer).to(device)
    optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
    num_update_steps_per_epoch = math.ceil(len(traindata) / (args.gradient_accumulation_steps * args.batch_size))
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    print("Start training MAIN MODEL")
    best_val_loss = 10000
    for epoch in range(args.num_train_epochs):
        model.train()
        model = train_one_epoch(
            args,
            epoch,
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            criterion,
            tokenizer=tokenizer,
        )
        model.eval()
        with torch.no_grad():
            val_loss = eval_one_epoch(
                args,
                model,
                valid_dataloader,
                criterion,
                tokenizer=tokenizer,
            )
        val_ppl = math.exp(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logging(f"MAIN MODEL Epoch {epoch} | Validation PPL: {val_ppl} | Learning rate: {current_lr}")
        # Save models
        save_checkpoint(model, tokenizer, args.outputdir, epoch)
        if val_loss < best_val_loss:
            logging(f"Saving best MAIN MODEL at Epoch {epoch}")
            save_checkpoint(model, tokenizer, args.outputdir, "best")
            best_val_loss = val_loss


def save_checkpoint(model, tokenizer, outputdir, epoch):
    fulloutput = os.path.join(outputdir, "checkpoint.{}".format(epoch))
    os.system(f"mkdir -p {fulloutput}")
    checkpoint = OrderedDict()
    # save tokenizer
    tokenizer.save_pretrained(fulloutput)
    # save configuration
    if model.llm.config._name_or_path == "gpt2":
        torch.save(model.llm.state_dict(), os.path.join(fulloutput, "pytorch_model.pt"))
    else:
        model.llm.save_pretrained(fulloutput)
    return checkpoint


def get_cascaded_uncertainty(model, prompt_nbest, generate_hyps, tokenizer, lengths):
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


def get_next_labelset(args, tokenizer, model, traindata):
    traindata.preprocess = False
    active_loader = DataLoader(traindata, batch_size=1, shuffle=False, collate_fn=collate_fn_active)
    firstpass_ids_dict = {}
    uncertainties = []
    for batch in tqdm(active_loader):
        slurp_ids, sequences, nbest = batch
        if args.asrplace == "weak" or args.asrplace == "both" and nbest[0] != []:
            firstpass_ids_dict[slurp_ids[0]] = []
            # nbest[0].append([tokenizer(sequences[0]).input_ids])  # Adding reference
            tokenized_seq = torch.tensor([nbest[0][0][0]]).to(device)
            outputs = model.generate_beam(
                tokenized_seq,
                max_new_tokens=64,
                beamsize=1,
            )
            lengths = torch.tensor([len(hyp.yseq) for hyp in outputs]).to(device)
            logplist = torch.stack([hyp.cumscore for hyp in outputs])
            predictive_entropy = get_cascaded_uncertainty(model, nbest[0], outputs, tokenizer, lengths)
            uncertainties.append(predictive_entropy.item())
            firstpass_ids_dict[slurp_ids[0]].append([tokenizer.decode(outputs[0].yseq).split("</s>")[0], predictive_entropy])
        else:
            tokenized_seq = tokenizer(sequences[0], return_tensors="pt").input_ids.to(device)
            outputs = model.generate_beam(
                tokenized_seq,
                max_new_tokens=64,
                beamsize=5,
            )
            lengths = torch.tensor([len(hyp.yseq) for hyp in outputs]).to(device)
            # if nbest[0] != []:
            #     predictive_entropy = get_cascaded_uncertainty(model, nbest[0], outputs, tokenizer, lengths)
            # else:
            logplist = torch.stack([hyp.cumscore for hyp in outputs])
            predictive_entropy, unnorm_entropy, _ = calc_predictive_entropy(logplist, 1, lengths)
            logplist = (- logplist / lengths).tolist()
            uncertainties.extend(logplist)
            firstpass_ids_dict[slurp_ids[0]] = [[tokenizer.decode(output.yseq).split("</s>")[0], logplist[i]] for i, output in enumerate(outputs)]
    uncertainties = sorted(uncertainties, reverse=True)
    threshold = uncertainties[int(args.unc_threshold * len(uncertainties))] if args.unc_threshold >= 0 else args.unc_threshold
    logging(f"Threshold for uncertainty: {threshold}")
    traindata.update_with_firstpass(firstpass_ids_dict, threshold)
    traindata.preprocess = True
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
        inputs, labels, nbest_prompt = batch
        with torch.cuda.amp.autocast():
            output, labels, knowledgeoutput = model(
                inputs,
                labels,
                knowledge=knowledge,
            )
            logits = output.logits
            seplosses = None
            loss = criterion(logits.view(-1, logits.size(-1)), labels.reshape(-1))
            loss = loss / args.gradient_accumulation_steps
        loss.backward()

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


def eval_one_epoch(args, model, valid_dataloader, criterion, knowledge=None, tokenizer=None):
    total_tokens = 0
    total_loss = 0.
    total_kgloss = 0.
    total_kgtokens = 0
    for i, batch in enumerate(valid_dataloader):
        inputs, labels, nbest_prompt = batch
        with torch.cuda.amp.autocast():
            output, labels, knowledgeoutput = model(
                inputs,
                labels,
                knowledge=knowledge,
            )
            logits = output.logits
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        tokens = (labels != -1).sum()
        total_tokens += tokens
        total_loss += loss.item() * tokens
    val_loss = total_loss / total_tokens
    return val_loss


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
        "--weak_model_path",
        type=str,
        default="gpt2",
        help="Path to the weak supervisor model file",
    )
    parser.add_argument(
        "--weak_train_samples",
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
        "--use_lora",
        type=str,
        default="false",
        help="Use lora for finetuning",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="slot",
        help="Use ptr for finetuning",
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
        "--num_candidates",
        type=int,
        default=1,
        help="Size of ensemble",
    )
    args = parser.parse_args()
    main(args)
