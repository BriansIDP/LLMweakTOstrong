# Weak to Strong Alignment for Slot Filling

## Requirements
torch==2.0.1 \
transformers==4.35.0 \
peft==0.3.0


## Training
Use `train.sh` to run training

Note that we use "weaker model" to refer to the weaker weak model, and "stronger model" to refer to the stronger weak model. "Main" refers to the main LLM.

Some specific training params:

`--model_path`: Main model name or path \
`--weak_model_path`: Weaker model name or path \
`--strong_model_path`: Stronger model name or path \
`--pretrained_weak_model_path`: Pretrained weaker model name or path. If the first time, use "none" \
`--pretrained_strong_model_path`: Pretrained stronger model name or path. If the first time, use "none" \
`--weak_train_samples`: Number of samples for weaker model, typically 500 or 1000 if stronger model uses 2000 \
`--strong_train_samples`: Number of samples for stronger model \
`--train_data_path`: Path to the training data for the main LLM \
`--weak_train_path`: Path to the training data for weaker and stronger models \
`--val_data_path`: Path to the validation file \
`--use_lora`: Whether to use LoRA \
`--ontology`: Not used. Keep this setting \
`--KBdrop`: Not used. Keep this setting \
`--maxKBsize`: Not used. Keep this setting \
`--lora_config`: Config file path for LoRA \
`--task`: Choose from: "normal", "deliberation" and "human_annotation". "normal" uses stronger model labels without deliberation. "human_annotation" uses human labels. \
`--asrplace`: Where to use ASR hypotheses, choose from: "both", "none", "weak", "main". Just keep "none" for now. \
`--num_candidates`: Weaker model may generate more than 1 hypotheses, so we can use multiple of those as input to deliberate. \

## Inference
Use `eval.sh` for inference

Some infernece params:

`--model_path`: Give the exp dir which contains weaker model, stronger model and main LLM checkpoints \
`--main_ckpt`: Which checkpoint to use for the main LLM \
`--recogfile`: Test data file containing reference and ASR output utterances \
`--topn`: Keep 1 for now \
`--samples`: Use 2000 sample setting for now \
`--asrname`: Keep medium for now - Whisper medium model output \
`--maxKBsize`: Not used. Keep this setting \
`--calibration_t`: Not used. Keep this setting  \
`--tag`: Tagging appended at the end of output file name to distinguish between inference runs. "upperbound" means using reference rather than ASR hyps. \
`--iteration`: Number of deliberation iterations. We can have more iterations by feeding the output back to the input to do another round of deliberation which leads to marginal improvements. \

## Scoring
`cd scoring` \
`python process.py <path_to_your_inference_output>`, e.g. ../exp/SLURP_w2s/SLURP_gpt2_vicuna7b_2000_samples_weak500to2000/output_medium_top1upperbound_iter1.json \
You will get one .jsonl file that can be scored using the official SLURP evaluation. \

`cd evaluation` \
`bash eval.sh`: Note you need to modify the path after `-p` \

## Train and Infer (for w2s w/o delibration)
train_weak_to_strong_clear.py: clean w2s code
train_weak.sh: Weak model training
train.sh: Strong model training
trian_search.sh: W2S training, searching seed and loss

infer_single.py: Infer of single model
eval.sh: Evaluate

## Config for w2s with soft label
task=multi_weak (both single weak and multi weaks)
strong_score_wordpiece=True
weak_model_names: target weak model (one for single model and numbers for multi models).