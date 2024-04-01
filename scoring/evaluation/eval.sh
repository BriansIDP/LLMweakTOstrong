. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
# python evaluate.py -g ref_2000_zeroshot.jsonl -p ../../exp/SLURP_ensemble/SLURP_vicuna7bv1.5_2000_samples_zeroshot_lora_qko/output_medium_top1_uncertainty_entropy_cascaded_onebest.jsonl \
#     --heldout "podcast_name artist_name audiobook_name business_name radio_name" \
# python evaluate.py -g ../testcleanslot.jsonl -p ../../exp/SLURP_w2s/SLURP_vicuna7b_vicuna7b_2000_samples_weak1000sample_uncertainty/output_medium_top1iter1.jsonl \
python evaluate.py -g ../testcleanslot.jsonl -p ../../exp/SLURP_w2s/SLURP_gpt2_vicuna7b_5000_samples_weak1000to5000/output_medium_top1upperbound_iter1.jsonl \
# --heldout  "podcast_name artist_name audiobook_name business_name radio_name" \
# python evaluate.py -g testcleanslot.jsonl --freqfile /home/gs534/rds/hpc-work/work/slurp/dump/train/deltafalse/entity_freqs.json -p /home/gs534/rds/hpc-work/work/transformers/examples/pytorch/language-modeling/gpt2_output/slurp_conformer_sepslottcpgen20_10distractor_noproj_nomask_sche15_copysup_GCNslot3in_f30_maskedless_share/decode_test_no_lm_b30_KBontof30_topclass2_classptr_sepslots/prediction.jsonl 
# python evaluate.py -g testunseen.jsonl -p /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/slurp/exp/slurp_conformer_GCNslot2_finetuneLibri_fullslotgen_10distractor_jointrepfull_maskedless/decode_test_no_lm_b30_KBontof30_topclass2_classptr/prediction.jsonl
#python evaluate.py \
#    -g testremix.jsonl \
#    -p /home/gs534/rds/hpc-work/work/transformers/examples/pytorch/language-modeling/gpt2_output/SLURP_gpt2_wordLM0.0_intent0.0_slot1.0_pairs_remix_parallel/predtest.jsonl \
#    --heldout "podcast_name artist_name audiobook_name business_name radio_name" \
# python split_and_evaluate.py \
#     -g testcleanslot.jsonl \
#     -p1 /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/slurp/exp/slurp_full_conformer_finetuneLibri_lstmproj_slotmix_mmatchfull_jointrep/decode_test_no_lm_b30/prediction.jsonl \
#     -p2 /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/slurp/exp/slurp_full_conformer_joint_GCN_randomKBslot2_f30_slotmix_mmatch_sampleword_classpost_jointrep2/decode_test_no_lm_b30_KBontof30_topclass2_post0.5/prediction.jsonl


# Evaluation for ChatGPT
# python evaluate.py \
#     -g testcleanslot.jsonl \
#     -p /home/gs534/rds/hpc-work/work/transformers/examples/pytorch/language-modeling/SLURP_data/chatgpt_response/testchatgpt_unbounded_full.jsonl \
#     --heldout "podcast_name" \

