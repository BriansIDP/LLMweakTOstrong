# . /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
python evaluate.py -g ../testcleanslot.jsonl -p ../../exp/debug/output_medium_top1upperbound_iter1.jsonl \
