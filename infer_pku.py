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

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteriaList, StoppingCriteria
from transformers.modeling_utils import load_sharded_checkpoint
from peft import PeftModel, PeftConfig

from knowledgemodel import KnowledgeLLM
from scoring.evaluation.normalizers.english import EnglishTextNormalizer

normaliser = EnglishTextNormalizer()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

PROMPT = "USER: {}\nASSISTANT: "


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


def main(args):
    start_time = time.time()
    with open(os.path.join(args.model_path, "model_config.json")) as fin:
        train_args = json.load(fin)

    ## Initialise tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_args["model_path"], use_fast=("pythia" in train_args["model_path"] or "bloom" in train_args["model_path"]))

    # Stopping criterion
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops='</s>', tokenizer=tokenizer)])

    # determine model type
    LLMtype = "vicuna"
    # if "llama-2" in train_args["model_path"]:
    #     LLMtype = "llama2"

    llm = AutoModelForCausalLM.from_pretrained(
        train_args["model_path"],
        # torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # if "gpt2" not in train_args["model_path"]:
    #     # config = PeftConfig.from_pretrained(peftpath)
    #     llm = PeftModel.from_pretrained(llm, os.path.join(args.model_path, args.main_ckpt), adapter_name="ada_1")
    # else:
    #     state_dict = torch.load(os.path.join(args.model_path, args.main_ckpt, "pytorch_model.pt"))
    #     llm.load_state_dict(state_dict)

    if os.path.exists(os.path.join(args.model_path, args.main_ckpt)):
        if train_args["use_lora"] == 'true':
            llm = PeftModel.from_pretrained(llm, os.path.join(args.model_path, args.main_ckpt), adapter_name="ada_1")
        elif os.path.exists(os.path.join(args.model_path, args.main_ckpt, "pytorch_model.pt")):
            state_dict = torch.load(os.path.join(args.model_path, args.main_ckpt, "pytorch_model.pt"))
            llm.load_state_dict(state_dict)
        elif os.path.exists(os.path.join(args.model_path, args.main_ckpt, "model.safetensors")):
            # state_dict = load_state_dict(os.path.join(args.model_path, args.main_ckpt, "model.safetensors"))
            # llm.load_state_dict(state_dict)
            llm = AutoModelForCausalLM.from_pretrained(
                os.path.join(args.model_path, args.main_ckpt),
                # torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            load_sharded_checkpoint(llm, os.path.join(args.model_path, args.main_ckpt))
    else:
        raise ValueError("Checkpoint don't exist.")

    model = KnowledgeLLM(llm, tokenizer, train_args["use_lora"])
    model.eval()

    # Read test file
    with open(args.recogfile) as fin:
        utterances = json.load(fin)

    start = time.time()
    outputdict = {}
    outputlist = []
    count = 0

    with torch.no_grad():
        for utt in utterances:
            prompt_txt = utt["prompt"]
            prompt = PROMPT.format(prompt_txt)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.llm.device)
            generate_hyps = model.generate_beam(
                input_ids=inputs.input_ids,
                max_new_tokens=256,
                stopping_criteria=stopping_criteria,
                beamsize=3,
                n_adapters=1,
            )
            # output = [tokenizer.decode(hyp.yseq).split("</s>")[0] for hyp in generate_hyps]
            output = [tokenizer.decode(hyp.yseq) for hyp in generate_hyps]

            # Get outputs
            outputs = model.tokenizer.batch_decode([generate_hyps[0].yseq], skip_special_tokens=True, clean_up_tokenization_spaces=False)

            outputlist.append([prompt_txt, outputs[0]])
            # outputdict[slurpid] = {
            #     "output": output,
            #     "predictive": predictive_entropy.item(),
            #     "unnormalised": unnorm_entropy.item(),
            #     "entity_f1": entity_f1,
            #     "slu_f1": slu_f1,
            #     "beamsearch_entropy": 0,
            #     "seg_correctness": 0,
            #     "seg_uncertainty": 0,
            # }
            count += 1
            logging("Finished {}, Elapsed time {:.2f}".format(count, time.time()-start_time))

    with open(os.path.join(args.model_path, args.result_file), "w") as fout:
        json.dump(outputlist, fout, indent=4)
            


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
