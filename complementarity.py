import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import wer

weak_model_list = ["gpt2-large", "opt-1.3b", "pythia-1.4b", "bloom-1b1", "TinyLlama_v1.1"]

confusion_matrix = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]

for i, weak_model in enumerate(weak_model_list):
    for j, ref_weak_model in enumerate(weak_model_list):
        wer_list = []
        result_file_1 = f"exp/weak/{weak_model}/output_medium_top1upperbound_iter1.json"
        result_file_2 = f"exp/weak/{ref_weak_model}/output_medium_top1upperbound_iter1.json"
        result_set_1 = json.load(open(result_file_1, encoding='utf-8'))
        result_set_2 = json.load(open(result_file_2, encoding='utf-8'))
        
        for key in result_set_1:
            result_1 = result_set_1[key]["output"]
            result_2 = result_set_2[key]["output"]
            error1 = wer(result_1, result_2)
            error2 = wer(result_2, result_1)
            # confusion_matrix[i][j] = (error1 + error2) / 2
            wer_list.append((error1 + error2) / 2)
        confusion_matrix[i][j] = sum(wer_list) / len(wer_list)

print(confusion_matrix)

# confusion_matrix = np.array(confusion_matrix)

# plt.figure(figsize=(10, 7))
# sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
# class_labels = weak_model_list
# plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=45)
# plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=0)

# plt.show()