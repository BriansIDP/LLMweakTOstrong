import pickle
import json
import numpy as np
from scipy.stats import binomtest

result_file_1 = "exp/w2s_corr_weak/gop_to_llama2/multi/lr1e-5_bs1*2_epoch2_soft_step_1/output_medium_top1upperbound_iter1.json"
result_file_2 = "exp/w2s_corr_weak/gop_to_llama2/edl/lr1e-5_bs1*2_epoch2_edl_rescale_ss2_weight_step_fordpo/output_medium_top1upperbound_iter1.json"
result_set_1 = json.load(open(result_file_1, encoding='utf-8'))
result_set_2 = json.load(open(result_file_2, encoding='utf-8'))
# print(pickle.load(open(result_file_1, 'rb'), encoding='utf-8')["avg_acc_test"], pickle.load(open(result_file_2, 'rb'), encoding='utf-8')["avg_acc_test"])

win_num = 0
lose_num = 0
equal_num = 0
# for i in range(len(result_set_1)):
#     result_1 = result_set_1[i]
#     result_2 = result_set_2[i]
#     assert (result_1["txt"] == result_2["txt"])
#     correct_1 = (result_1["gt_label"] == result_1["hard_label"])
#     correct_2 = (result_2["gt_label"] == result_2["hard_label"])
#     if (correct_1 and not correct_2):
#         win_num += 1
#     elif (not correct_1 and correct_2):
#         lose_num += 1
#     else:
#         equal_num += 1
for key in result_set_1:
    result_1 = result_set_1[key]["slu_f1"]
    result_2 = result_set_2[key]["slu_f1"]
    if result_1 > result_2:
        win_num += 1
    elif result_1 < result_2:
        lose_num += 1
    else:
        equal_num += 1
print(win_num, lose_num, equal_num)
print(win_num + lose_num + equal_num)

n = win_num + lose_num
p_value = binomtest(win_num, n=n, p=0.5, alternative='greater')

print("P-value for the binomial test of superiority:", p_value.pvalue)