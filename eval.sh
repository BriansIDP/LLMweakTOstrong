. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate llama

# python get_knowledge_encodings.py --model_name vicuna-7b-v1.5-16k
asrname="medium"
# asrfile="data/${asrname}_nbest_zeroshot.json"
asrfile="data/${asrname}_2000.json"
nsamples=2000
# expdir="exp/SLURP_w2s/SLURP_vicuna7bASR_vicuna7b_${nsamples}_samples_weak1000sample"
expdir="exp/SLURP_w2s/SLURP_gpt2_vicuna7bASR_${nsamples}_samples_weak1000sample_uncertainty_value"
# expdir="exp/SLURP/SLURP_vicuna13bv1.5_${nsamples}_samples_zeroshot_baseline2"
logfile="$expdir/eval_log.txt"
python inference.py \
    --model_path $expdir \
    --main_ckpt checkpoint.best \
    --recogfile $asrfile \
    --topn 1 \
    --samples ${nsamples} \
    --asrname ${asrname} \
    --logfile $logfile \
    --ontology data/ontology_norm.json \
    --maxKBsize 0 \
    --calibration_t 1 \
    --tag iter1 \
    --iteration 1 \
    # --unc_threshold -1 \
    # --cascaded \
    # --ckptlist exp/checkpoints.txt \
    # --do_sampling \
    # --cutoff_prob 0.7 \
    # --knowledge_embs data/knowledge/knowledge_slot_vicuna-7b-v1.5.pt \
