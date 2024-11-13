# . /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate llama

# python get_knowledge_encodings.py --model_name vicuna-7b-v1.5-16k
asrname="medium"
# asrfile="data/${asrname}_nbest_zeroshot.json"
asrfile="data/${asrname}_2000.json"
nsamples=2000
# expdir="exp/SLURP_w2s/SLURP_vicuna7bASR_vicuna7b_${nsamples}_samples_weak1000sample"
# expdir="exp/SLURP_w2s/SLURP_gpt2_vicuna7b_${nsamples}_samples_weak500to2000"
# expdir="exp/weak/joint_decode_5+3+2"
expdir="exp/w2s_corr_weak/gopbt_to_llama2/cdpo/edlstep_seed2_beta5_lr5e-7_sum_10top_3epoch_ls0.1_weakfile"
# expdir="exp/SLURP/SLURP_vicuna1\3bv1.5_${nsamples}_samples_zeroshot_baseline2"
logfile="$expdir/eval_log.txt"

num_list="0 1 2"
for num in $num_list
do
    result_file="output_${asrname}_top1upperbound_iter1_${num}.json"

    if [ -e "${expdir}/${result_file}" ]; then
        echo "Infer has been done before"
    else
        CUDA_VISIBLE_DEVICES=6 \
        python infer_single.py \
            --model_path $expdir \
            --main_ckpt checkpoint.${num} \
            --recogfile $asrfile \
            --result_file $result_file \
            --topn 1 \
            --samples ${nsamples} \
            --asrname ${asrname} \
            --logfile $logfile \
            --ontology data/ontology_norm.json \
            --maxKBsize 0 \
            --calibration_t 1 \
            --tag upperbound_iter1 \
            --iteration 1 \
            > ${expdir}/eval_${num}.log 2>&1
    fi
        # --unc_threshold 0.06 \
        # --cascaded \s
        # --ckptlist exp/checkpoints.txt \
        # --do_sampling \
        # --cutoff_prob 0.7 \
        # --knowledge_embs data/knowledge/knowledge_slot_vicuna-7b-v1.5.pt \

    cd ./scoring
    result_file1="${result_file}l"
    if [ -e "../${expdir}/${result_file1}" ]; then
        echo "Process has been done before"
    else
        python process.py ../${expdir}/${result_file}
    fi

    cd ./evaluation
    python evaluate.py -g ../testcleanslot.jsonl -p ../../${expdir}/${result_file1} > ../../${expdir}/eval_result_${num}.log

    cd ../../
done