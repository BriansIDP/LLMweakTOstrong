# . /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate videollama

nsample=2000
weaksample=2000
strongsample=2000

# trainfile=data/trainlabel_nbest_debug.json
trainfile=data/pkusafe/train_data_2000.json
# trainfile=data/validlabel_nbest.json

trainweakfile=data/pkusafe/train_weak_data_2000.json

# valfile=data/trainlabel_nbest_debug.json
valfile=data/pkusafe/valid_data.json


# weakmodel=ckpt/vicuna-7b-v1.5
# weakmodel=/mnt/nvme_share/cuizy/models/gpt2-large
weakmodel=/mnt/nvme_share/cuizy/models/pythia-1.4b
# weakmodel=/mnt/nvme_share/cuizy/models/opt-1.3b
# weakmodel=/mnt/nvme_share/cuizy/models/bloom-560m
# modelpath=/mnt/nvme_share/cuizy/models/llama-2-7b-hf
# modelpath=/mnt/nvme_share/cuizy/models/llama-2-7b-chat-hf
modelpath=/mnt/nvme_share/cuizy/models/vicuna-7b-v1.1
# modelpath=/mnt/nvme_share/cuizy/models/gpt2-large
# modelpath=/mnt/nvme_share/cuizy/models/opt-1.3b
# modelpath=/mnt/nvme_share/cuizy/models/pythia-1.4b
# modelpath=/mnt/nvme_share/cuizy/models/bloom-560m


# pretrained_weak_model_path=exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}/checkpoint.best_weak
# pretrained_strong_model_path=exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}/checkpoint.best_stronger
# pretrained_weak_model_path="exp/weak/gpt2-large/checkpoint.best"
pretrained_weak_model_path="exp_pku/weak/pythia-1.4b_lr5e-5_bs4*2/checkpoint.best"
# pretrained_weak_model_path="exp_pku/weak/opt-1.3b_lr1e-5_bs4*2/checkpoint.best"


# expdir="exp_pku/strong/vicuna-v1.1_lr1e-5"
expdir="exp_pku/w2s/gop_to_vicuna/edl/lr3e-6_bs1*2_edl_step"
mkdir -p $expdir

CUDA_VISIBLE_DEVICES=2 \
python train_weak_to_strong_pku.py \
    --model_path $modelpath \
    --weak_model_path $weakmodel \
    --strong_model_path $weakmodel \
    --pretrained_weak_model_path $pretrained_weak_model_path \
    --weak_train_samples $weaksample \
    --strong_train_samples $strongsample \
    --batch_size 1 \
    --eval_batch_size 8 \
    --learning_rate 3e-6 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 50 \
    --train_data_path $trainfile \
    --weak_train_path $trainweakfile \
    --val_data_path $valfile \
    --use_lora false \
    --ontology data/ontology_norm.json \
    --KBdrop 0.0 \
    --maxKBsize 0 \
    --lora_config data/lora_config.json \
    --task multi_weak \
    --criterion edl_step \
    --weak_model_names opt-1.3b_lr5e-5_bs4*2,gpt2-large_lr5e-5_bs4*2,pythia-1.4b_lr5e-5_bs4*2 \
    --asrplace none \
    --num_candidates 1 \
    --strong_score_wordpiece \
    > ${expdir}/train_log.log 2>&1

    # --pretrained_weak_model_path $pretrained_weak_model_path \
    # --pretrained_strong_model_path $pretrained_strong_model_path \
    # --criterion logconf \
    # --unc_threshold 0.5 \
    # --selflabelling \
    # --topn 10 \
    # --strong_score_wordpiece \
    # --weak_model_names gpt2-large,opt-1.3b,pythia-1.4b \
