asrname="medium"
# asrfile="data/${asrname}_nbest_zeroshot.json"
# asrfile="data/pkusafe/eval_data.json"
asrfile="data/pkusafe/test_data_1000.json"

expdir="exp_pku/w2s/gop_to_vicuna/edl/lr1e-5_bs1*2_edl_step"

logfile="$expdir/eval_log.txt"
# result_file="infer_BeaverTails-Evaluation.json"
result_file="infer_safeRLHF10k.json"
device=2

if [ -e "${expdir}/${result_file}" ]; then
    echo "Infer has been done before"
else
    CUDA_VISIBLE_DEVICES=$device \
    python infer_pku.py \
        --model_path $expdir \
        --main_ckpt checkpoint.1 \
        --recogfile $asrfile \
        --result_file $result_file \
        --topn 1 \
        --samples 700 \
        --asrname ${asrname} \
        --logfile $logfile \
        --ontology data/ontology_norm.json \
        --maxKBsize 0 \
        --calibration_t 1 \
        --tag upperbound_iter1 \
        --iteration 1
fi

CUDA_VISIBLE_DEVICES=$device \
python eval_pku_cost.py \
    --target_folder $expdir \
    --json_file $result_file \
    > ${expdir}/eval_result_test.log 2>&1