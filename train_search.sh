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
# weakmodel=/mnt/nvme_share/cuizy/models/gpt2-large
weakmodel=/mnt/nvme_share/cuizy/models/pythia-1.4b
# weakmodel=/mnt/nvme_share/cuizy/models/opt-1.3b
# weakmodel=/mnt/nvme_share/cuizy/models/bloom-1b1
# weakmodel=/mnt/nvme_share/cuizy/models/TinyLlama_v1.1

modelpath=/mnt/nvme_share/cuizy/models/llama-2-7b-hf
# modelpath=/mnt/nvme_share/cuizy/models/gpt2-large
# modelpath=/mnt/nvme_share/cuizy/models/opt-1.3b
# modelpath=/mnt/nvme_share/cuizy/models/pythia-1.4b
# modelpath=/mnt/nvme_share/cuizy/models/bloom-1b1
# modelpath=/mnt/nvme_share/cuizy/models/TinyLlama_v1.1

# pretrained_weak_model_path="exp/weak/gpt2-large/checkpoint.best"
# pretrained_weak_model_path="exp/weak/opt-1.3b/checkpoint.best"
pretrained_weak_model_path="exp/weak/pythia-1.4b/checkpoint.best"
# pretrained_weak_model_path="exp/weak/bloom-1b1/checkpoint.best"
# pretrained_weak_model_path="exp/weak/TinyLlama_v1.1/checkpoint.best"

# loss_list="logconf_step"
# loss_list="soft soft_step"
loss_list="soft"
# loss_list="edl_step"
seed_list="2"
for seed in $seed_list
do
for loss in $loss_list
do
    # expdir="exp/w2s_corr_weak/gop_to_llama2/back/lr1e-5_bs1*2_epoch2_${loss}_hard_check_${num}"
    expdir="exp/w2s_corr_weak/gop_to_llama2/joint/lr1e-5_bs1*2_epoch2_${loss}_3best_seed${seed}"
    mkdir -p $expdir
    cp train_search.sh $expdir

    CUDA_VISIBLE_DEVICES=6 \
    python train_weak_to_strong_clear.py \
        --model_path $modelpath \
        --weak_model_path $weakmodel \
        --strong_model_path $weakmodel \
        --pretrained_weak_model_path $pretrained_weak_model_path \
        --weak_train_samples $weaksample \
        --strong_train_samples $strongsample \
        --batch_size 1 \
        --eval_batch_size 8 \
        --learning_rate 1e-5 \
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
        --task joint_decode \
        --criterion $loss \
        --weak_model_names gpt2-large,opt-1.3b,pythia-1.4b \
        --asrplace none \
        --num_candidates 1 \
        --aux_coef 0.2 \
        --seed $seed \
        --nbest 3 \
        --strong_score_wordpiece \
        > ${expdir}/train_log.log 2>&1
done
done
# --strong_score_wordpiece \
# --weak_model_names gpt2-large_1,opt-1.3b_1,pythia-1.4b_1 \
# --data_pre_generated \
# --weak_model_names gpt2-large,opt-1.3b,pythia-1.4b,bloom-1b1,TinyLlama_v1.1 \