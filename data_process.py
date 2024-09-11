import json
origin_file = "/mnt/nvme_share/cuizy/LLMweakTOstrong/exp/weak/joint_decode_oracle/output_medium_top1upperbound_iter1.json"
target_file = "/mnt/nvme_share/cuizy/LLMweakTOstrong/exp/weak/joint_decode_oracle/output_medium_top1upperbound_iter1_1.json"
with open(origin_file) as f:
    output_dic = json.load(f)
for key in output_dic:
    output_dic[key]["output"] = output_dic[key]["output"][0]
with open(target_file, 'w') as f:
    json.dump(output_dic, f, indent=4)