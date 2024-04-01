# Weak to Strong Alignment for Slot Filling

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
