# . /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate videollama

nsample=2000
weaksample=2000
strongsample=2000
# expdir="exp/SLURP_active/SLURP_vicuna7bv1.5_${nsample}_samples_qko_reset200_b5"
# expdir="exp/SLURP_w2s/SLURP_vicuna7b_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}"
# expdir="exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}_humanannot_allparam"

# trainfile=data/trainlabel_nbest_debug.json
trainfile=data/trainlabel_nbest_${nsample}.json
# trainfile=data/validlabel_nbest.json

trainweakfile=data/trainlabel_exclusive_${strongsample}.json

# valfile=data/trainlabel_nbest_debug.json
valfile=data/validlabel_nbest.json


# weakmodel=ckpt/vicuna-7b-v1.5
weakmodel=/mnt/nvme_share/cuizy/models/gpt2-large
# weakmodel=/mnt/nvme_share/cuizy/models/pythia-1.4b
# weakmodel=/mnt/nvme_share/cuizy/models/opt-1.3b
# weakmodel=/mnt/nvme_share/cuizy/models/bloom-560m
# modelpath=/mnt/nvme_share/cuizy/models/llama-2-7b-hf
modelpath=/mnt/nvme_share/cuizy/models/gpt2-large
# modelpath=/mnt/nvme_share/cuizy/models/opt-1.3b
# modelpath=/mnt/nvme_share/cuizy/models/pythia-1.4b
# modelpath=/mnt/nvme_share/cuizy/models/bloom-560m


# pretrained_weak_model_path=exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}/checkpoint.best_weak
# pretrained_strong_model_path=exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}/checkpoint.best_stronger
# pretrained_weak_model_path="exp/weak/gpt2-large/checkpoint.best"
# pretrained_weak_model_path="exp/weak/opt-1.3b/checkpoint.best"
pretrained_weak_model_path="exp/weak/pythia-1.4b/checkpoint.best"
pretrained_strong_model_path=exp/debug/checkpoint.best_stronger


expdir="exp/weak/gpt2-large_padding=100_2"
mkdir -p $expdir

CUDA_VISIBLE_DEVICES=0 \
python train_weak_to_strong_clear.py \
    --model_path $modelpath \
    --weak_model_path $weakmodel \
    --strong_model_path $weakmodel \
    --weak_train_samples $weaksample \
    --strong_train_samples $strongsample \
    --batch_size 4 \
    --eval_batch_size 8 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 15 \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 50 \
    --train_data_path $trainweakfile \
    --weak_train_path $trainweakfile \
    --val_data_path $valfile \
    --use_lora false \
    --ontology data/ontology_norm.json \
    --KBdrop 0.0 \
    --maxKBsize 0 \
    --lora_config data/lora_config.json \
    --task human_annotation \
    --criterion xent \
    --weak_model_names gpt2-large,opt-1.3b,pythia-1.4b \
    --asrplace none \
    --num_candidates 1 \
    > ${expdir}/train_log.log

    # --pretrained_weak_model_path $pretrained_weak_model_path \
    # --pretrained_strong_model_path $pretrained_strong_model_path \
    # --criterion logconf \
    # --unc_threshold 0.5 \
    # --selflabelling \
    # --topn 10 \
