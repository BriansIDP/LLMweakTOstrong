. /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate videollama

nsample=5000
# expdir="exp/SLURP_active/SLURP_vicuna7bv1.5_${nsample}_samples_qko_reset200_b5"
expdir="exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsample}_samples_weak500to5000_normal"
# expdir="exp/SLURP/SLURP_llama13b_${nsample}_nbest_samples_zeroshot"

# trainfile=data/trainlabel_norm_${nsample}.json
trainfile=data/trainlabel_nbest_${nsample}.json
# trainfile=data/validlabel_nbest.json
trainweakfile=data/trainlabel_exclusive_5000.json
# valfile=data/validlabel_norm.json
valfile=data/validlabel_nbest.json

# expdir="exp/debug"
mkdir -p $expdir
python train_weak_to_strong.py \
    --model_path ckpt/vicuna-7b-v1.5 \
    --weak_model_path gpt2 \
    --strong_model_path gpt2 \
    --weak_train_samples 500 \
    --strong_train_samples 5000 \
    --batch_size 3 \
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
    --task normal \
    --asrplace none \
    --num_candidates 1 \
    # --criterion logconf \
    # --unc_threshold 0.5 \
    # --selflabelling \
    # --topn 10 \
    # --tag LKI \

    # ckpt/vicuna-7b-v1.5 \
    # ckpt/llama-2-13b-hf \
    # --model_path ckpt/vicuna-7b-v1.5 \
    # --model_path ckpt/vicuna-13b-v1.5/ \
    # --knowledge_embs data/knowledge/knowledge_slot_vicuna-7b-v1.5.pt \
    # --resume exp/slurp_llama13b_baseline_25000_samples_noschema_allparam2/checkpoint.ep.18 \
    # --use_attention true \
    # --nquery 1 \
