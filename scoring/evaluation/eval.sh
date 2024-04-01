. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
python evaluate.py -g ../testcleanslot.jsonl -p ../../exp/SLURP_w2s/SLURP_gpt2_vicuna7b_5000_samples_weak1000to5000/output_medium_top1upperbound_iter1.jsonl \
