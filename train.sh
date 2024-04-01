. /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate videollama

nsample=2000
weaksample=500
strongsample=2000
# expdir="exp/SLURP_active/SLURP_vicuna7bv1.5_${nsample}_samples_qko_reset200_b5"
# expdir="exp/SLURP_w2s/SLURP_vicuna7b_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}"
expdir="exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}_humanannot_allparam"

# trainfile=data/trainlabel_nbest_debug.json
trainfile=data/trainlabel_nbest_${nsample}.json
# trainfile=data/validlabel_nbest.json

trainweakfile=data/trainlabel_exclusive_${strongsample}.json

# valfile=data/trainlabel_nbest_debug.json
valfile=data/validlabel_nbest.json


# weakmodel=ckpt/vicuna-7b-v1.5
weakmodel=gpt2


pretrained_weak_model_path=exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}/checkpoint.best_weak
pretrained_strong_model_path=exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsample}_samples_weak${weaksample}to${strongsample}/checkpoint.best_stronger

# expdir="exp/debug"
mkdir -p $expdir
python train_weak_to_strong.py \
    --model_path ckpt/vicuna-7b-v1.5 \
    --weak_model_path $weakmodel \
    --strong_model_path $weakmodel \
    --pretrained_weak_model_path $pretrained_weak_model_path \
    --pretrained_strong_model_path $pretrained_strong_model_path \
    --weak_train_samples $weaksample \
    --strong_train_samples $strongsample \
    --batch_size 1 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 15 \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 50 \
    --train_data_path $trainfile \
    --weak_train_path $trainweakfile \
    --val_data_path $valfile \
    --use_lora true \
    --ontology data/ontology_norm.json \
    --KBdrop 0.0 \
    --maxKBsize 0 \
    --lora_config data/lora_config.json \
    --task human_annotation \
    --asrplace none \
    --num_candidates 1 \
    # --criterion logconf \
    # --unc_threshold 0.5 \
    # --selflabelling \
    # --topn 10 \
