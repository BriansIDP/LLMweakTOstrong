import random
from typing import Optional, Dict
import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

PROMPT = "USER: {}\nASSISTANT: "

class ActiveDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        prompts,
        labelset=[],
        LLM="vicuna",
        nbest=1,
        asrplace="none",
        num_candidates=1,
    ):
        super(ActiveDataset, self).__init__()
        self.asrplace = asrplace
        self.data = self.process_json_data(json.load(open(data_path)))
        self.labelset = labelset
        self.get_labelled_set(labelset)
        self.main_data = self.labelled
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.nbest = nbest
        self.LLM = LLM
        self.preprocess = True
        self.num_candidates = num_candidates
        self.multi_weak = False
        self.per_token_score = False
        self.strong_score = False

    def get_labelled_set(self, labelset):
        self.labelled = []
        self.unlabelled = []
        for slurpid, content in self.data.items():
            if slurpid in labelset:
                self.labelled.append(self.data[slurpid])
            else:
                self.unlabelled.append(self.data[slurpid])

    def refill_labelset(self, step=0):
        if step == 0:
            self.labelset = self.data.keys()
        else:
            self.labelset = random.sample(self.data.keys(), k=step)
        self.get_labelled_set(self.labelset)
        self.main_data = self.labelled

    def __len__(self):
        return len(self.main_data)

    def switch(self):
        if self.preprocess:
            self.main_data = self.unlabelled
            self.preprocess = False
        else:
            self.preprocess = True
            self.main_data = self.labelled

    def update(self, slurpids):
        self.labelset += slurpids
        self.labelled = []
        self.unlabelled = []
        for slurpid, content in self.data.items():
            if slurpid in self.labelset:
                self.labelled.append(self.data[slurpid])
            else:
                self.unlabelled.append(self.data[slurpid])

    def update_with_labels(self, labelset):
        self.labelled = []
        self.unlabelled = []
        covered_ids = []
        labelset = self.process_json_data(labelset)
        for slurpid, item in labelset.items():
            self.labelled.append(item)
            covered_ids.append(slurpid)
        for slurpid, content in self.data.items():
            self.unlabelled.append(self.data[slurpid])

    def update_with_firstpass(self, labelset):
        for datapiece in self.main_data:
            datapiece[2] = labelset[datapiece[0]][0][0]
            datapiece[3] = labelset[datapiece[0]][0][1]

    def update_with_firstpass_multiweak(self, labelset):
        for datapiece in self.main_data:
            labels, scores = zip(*(labelset[datapiece[0]]))
            datapiece[2] = list(labels)
            datapiece[3] = list(scores)

    def process_json_data(self, data):
        sludata = {}
        for datapiece in data:
            id = datapiece["id"]
            prompt = datapiece["prompt"]
            response = datapiece["response"]
            sludata[id] = [id, prompt, response, {}]
            # 0: slurp_id  1: input text  2: label  3: per word score
        return sludata

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if self.multi_weak:
            return self.preprocessing_multiweak(self.main_data[idx])
        else:
            return self.preprocessing(self.main_data[idx])

    def preprocessing(self, sample):
        slurpid, content, label, score = sample
        prompt = PROMPT.format(content)
        if self.preprocess:
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
            label_ids = self.tokenizer(label + "</s>", return_tensors="pt")["input_ids"]
            label_ids = label_ids[0, 1:] if label_ids[0, 0] == 1 else label_ids[0]
            total_ids = torch.cat([prompt_inputs["input_ids"][0], label_ids], dim=-1)
            total_label = torch.cat([prompt_inputs["input_ids"][0] * 0 - 1, label_ids], dim=-1)
            
            mapping = wp_word_map(
                wordpiece_list=[self.tokenizer.decode(id).replace(' ', '') for id in label_ids],
                word_list=(label+"</s>").split(' '),
            )
            if score == {}:
                token_score = {}
            elif score.numel() == 1:
                token_score = torch.full((len(label_ids), ), score.item() / len(label_ids))
            else:
                token_score = torch.zeros(len(label_ids))
                for i, (start, end) in enumerate(mapping):
                    token_score[start:end+1] = score[i] / (end-start+1)
            return total_ids, total_label, token_score
        else:
            return slurpid, prompt, label
        
    def preprocessing_multiweak(self, sample):
        slurpid, content, label_list, score_list = sample
        prompt = PROMPT.format(content)
        
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        total_ids_list = []
        total_label_list = []
        if not self.strong_score:
            token_score_list = []
            for label, score in zip(label_list, score_list):
                label_ids = self.tokenizer(label + "</s>", return_tensors="pt")["input_ids"]
                label_ids = label_ids[0, 1:] if label_ids[0, 0] == 1 else label_ids[0]
                total_ids = torch.cat([prompt_inputs["input_ids"][0], label_ids], dim=-1)
                total_label = torch.cat([prompt_inputs["input_ids"][0] * 0 - 1, label_ids], dim=-1)

                mapping = wp_word_map(
                    wordpiece_list=[self.tokenizer.decode(id).replace(' ', '') for id in label_ids],
                    word_list=(label+"</s>").split(' ')
                )
                if score == {}:
                    token_score = {}
                elif score.numel() == 1:
                    token_score = torch.full((len(label_ids), ), score.item() / len(label_ids))
                else:
                    token_score = torch.zeros(len(label_ids))
                    for i, (start, end) in enumerate(mapping):
                        token_score[start:end+1] = score[i] / (end-start+1)

                total_ids_list.append(total_ids)
                total_label_list.append(total_label)
                token_score_list.append(token_score)
            return total_ids_list, total_label_list, token_score_list
        
        elif self.strong_score:
            mapping_list = []
            for label, score in zip(label_list, score_list):
                label_ids = self.tokenizer(label + "</s>", return_tensors="pt")["input_ids"]
                label_ids = label_ids[0, 1:] if label_ids[0, 0] == 1 else label_ids[0]
                total_ids = torch.cat([prompt_inputs["input_ids"][0], label_ids], dim=-1)
                total_label = torch.cat([prompt_inputs["input_ids"][0] * 0 - 1, label_ids], dim=-1)

                mapping = wp_word_map(
                    wordpiece_list=[self.tokenizer.decode(id).replace(' ', '') for id in label_ids],
                    word_list=(label+"</s>").split(' ')
                )

                total_ids_list.append(total_ids)
                total_label_list.append(total_label)
                mapping_list.append(mapping)
            return total_ids_list, total_label_list, score_list, mapping_list


def collate_fn_active(batch):
    slurp_ids, sequences, label = zip(*batch)
    return slurp_ids, sequences, label


def collate_fn(batch):
    total_ids, total_label, scores = zip(*batch)

    total_ids = pad_sequence(total_ids, batch_first=True, padding_value=1).to(device)
    total_label = pad_sequence(total_label, batch_first=True, padding_value=-1).to(device)
    attn_mask = (total_ids != 1)
    inputs = {"input_ids": total_ids[:, :-1], "attention_mask": attn_mask[:, :-1]}

    if scores[0] != {}:
        scores = torch.concat(scores, dim=0)
    return inputs, total_label[:, 1:], scores


def collate_fn_multiweak(batch):
    '''
    Only works for batch_size=1 for now.
    '''
    total_ids, total_label, scores = zip(*batch)

    total_ids = pad_sequence(total_ids[0], batch_first=True, padding_value=1).to(device)
    total_label = pad_sequence(total_label[0], batch_first=True, padding_value=-1).to(device)
    attn_mask = (total_ids != 1)
    inputs = {"input_ids": total_ids[:, :-1], "attention_mask": attn_mask[:, :-1]}

    if scores[0][0] != {}:
        scores = torch.concat(scores[0], dim=0)
    return inputs, total_label[:, 1:], scores


def collate_fn_strongscore_mw(batch):
    '''
    task == multi_weak and strong_score_wordpiece == True
    '''
    total_ids_list, total_label_list, score_list, mapping_list = zip(*batch)
    total_ids_list = total_ids_list[0]
    total_label_list = total_label_list[0]
    score_list = score_list[0]
    mapping_list = mapping_list[0]
    inputs_list = []
    for i in range(len(total_ids_list)):
        # total_ids_list[i].unsqueeze(0)
        # total_label_list[i].unsqueeze(0)
        total_ids_list[i] = torch.unsqueeze(total_ids_list[i], dim=0)
        total_label_list[i] = torch.unsqueeze(total_label_list[i], dim=0)
        total_label_list[i] = total_label_list[i][:, 1:]
        attention_mask = total_ids_list[i] != 0
        inputs_list.append({"input_ids": total_ids_list[i][:, :-1], "attention_mask": attention_mask[:, :-1]})
    return inputs_list, total_label_list, score_list, mapping_list


def wp_word_map(wordpiece_list, word_list):
    '''
    Match wordpiece and word.
    Input: List[str] with correct order.
    Output: List[tuple(start, end) for word in word_list]
    '''
    mapping = []
    i = 0
    for word in word_list:
        start = i
        while i < len(wordpiece_list) and not word.endswith(wordpiece_list[i]):
            i += 1
        mapping.append((start, i))
        i += 1
    # mapping[-1][1] = len(wordpiece_list) - 1
    end_tuple = mapping[-1]
    if end_tuple[1] != len(wordpiece_list) - 1:
        mapping[-1] = (end_tuple[0], len(wordpiece_list) - 1)
    return mapping