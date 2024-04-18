import os
import re
import math
import pathlib
import random
from typing import Optional, Dict
from tqdm import tqdm
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data.prompt import templates


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ActiveDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        slots,
        slotstr,
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
        self.slotstr = slotstr
        self.slots = slots
        self.tokenizer = tokenizer
        self.nbest = nbest
        self.LLM = LLM
        self.preprocess = True
        self.num_candidates = num_candidates
        self.multi_weak = False

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
            # if slurpid not in covered_ids:
            self.unlabelled.append(self.data[slurpid])

    def update_with_firstpass(self, labelset, threshold, update_label=False, update_delib=False):
        for datapiece in self.main_data:
            if update_label:
                datapiece[2] = labelset[datapiece[0]][0][0]
                datapiece[3] = labelset[datapiece[0]][0][1]
            if update_delib:
                datapiece[4] = []
                for item in labelset[datapiece[0]][1:]:
                    datapiece[4].append(item[0])

    def update_with_firstpass_multiweak(self, labelset, threshold, update_label=True):
        for datapiece in self.main_data:
            labels, values = zip(*(labelset[datapiece[0]]))
            datapiece[2] = list(labels)
            datapiece[3] = list(values)
            # datapiece[2] = labelset[datapiece[0]][:, 0]
            # datapiece[3] = labelset[datapiece[0]][:, 1]

    def process_json_data(self, data):
        sludata = {}
        for utterance in data["data"]:
            label = {}
            random.shuffle(utterance["entities"])
            for ent in utterance["entities"]:
                if ent["type"] in label:
                    label[ent["type"]] += " & " + ent["value"]
                else:
                    label[ent["type"]] = ent["value"]

            labelstr = json.dumps(label)
            nbest = []
            if "nbest" in utterance and len(utterance["nbest"]) > 0:
                nbest = utterance["nbest"]  # random.choice(utterance["nbest"])
            sludata[utterance["slurp_id"]] = [utterance["slurp_id"], utterance["text"], labelstr, {}, "", nbest]
        return sludata

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if self.multi_weak:
            return self.preprocessing_multiweak(self.main_data[idx])
        else:
            return self.preprocessing(self.main_data[idx])

    def preprocessing(self, sample):
        slurpid, content, label, values, keystrings, nbest = sample
        nbest = random.choice(nbest)  # randomly choose one audio output
        if self.asrplace == "both" or self.asrplace == "main":
            tmp_nbest = nbest + [[content]]
            content = random.choice(tmp_nbest)[0]
        nbest_prompt = []
        system = self.prompts["system"]
        taskdesc = self.prompts["task_description"].format(self.slotstr)
        query = self.prompts["query"]
        if keystrings != "" and isinstance(keystrings, list):
            num_options = random.choice(range(1, self.num_candidates+1))
            keystring = ", ".join(random.choices(keystrings, k=num_options))
            query = self.prompts["delib_query"].format(keystring) + query
        prompt = templates[self.LLM]["slot"][0].format(**locals())
        for each_hyp in nbest:
            content = each_hyp[0]
            nbest_prompt.append([self.tokenizer(templates[self.LLM]["slot"][0].format(**locals())).input_ids, each_hyp[1]])
        if self.preprocess:
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
            input_size = len(prompt_inputs["input_ids"][0]) - 5
            label_ids = self.tokenizer(label + "</s>", return_tensors="pt")["input_ids"]
            label_ids = label_ids[0, 1:] if label_ids[0, 0] == 1 else label_ids[0]
            total_ids = torch.cat([prompt_inputs["input_ids"][0], label_ids], dim=-1)
            total_label = torch.cat([prompt_inputs["input_ids"][0] * 0 - 1, label_ids], dim=-1)
            return total_ids, total_label, nbest_prompt, values
        else:
            return slurpid, prompt, nbest_prompt, label
        
    def preprocessing_multiweak(self, sample):
        slurpid, content, label_list, values, keystrings, nbest = sample
        nbest = random.choice(nbest)  # randomly choose one audio output
        if self.asrplace == "both" or self.asrplace == "main":
            tmp_nbest = nbest + [[content]]
            content = random.choice(tmp_nbest)[0]
        nbest_prompt = []
        system = self.prompts["system"]
        taskdesc = self.prompts["task_description"].format(self.slotstr)
        query = self.prompts["query"]
        if keystrings != "" and isinstance(keystrings, list):
            num_options = random.choice(range(1, self.num_candidates+1))
            keystring = ", ".join(random.choices(keystrings, k=num_options))
            query = self.prompts["delib_query"].format(keystring) + query
        prompt = templates[self.LLM]["slot"][0].format(**locals())
        for each_hyp in nbest:
            content = each_hyp[0]
            nbest_prompt.append([self.tokenizer(templates[self.LLM]["slot"][0].format(**locals())).input_ids, each_hyp[1]])
        
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        total_ids_list = []
        total_label_list = []
        for label in label_list:
            label_ids = self.tokenizer(label + "</s>", return_tensors="pt")["input_ids"]
            label_ids = label_ids[0, 1:] if label_ids[0, 0] == 1 else label_ids[0]
            total_ids = torch.cat([prompt_inputs["input_ids"][0], label_ids], dim=-1)
            total_label = torch.cat([prompt_inputs["input_ids"][0] * 0 - 1, label_ids], dim=-1)
            total_ids_list.append(total_ids)
            total_label_list.append(total_label)
        return total_ids_list, total_label_list, nbest_prompt, values



def collate_fn_active(batch):
    slurp_ids, sequences, nbest_prompt, label = zip(*batch)
    return slurp_ids, sequences, nbest_prompt, label


def collate_fn(batch):
    total_ids, total_label, nbest, values = zip(*batch)

    total_ids = pad_sequence(total_ids, batch_first=True, padding_value=1).to(device)
    total_label = pad_sequence(total_label, batch_first=True, padding_value=-1).to(device)
    attn_mask = total_ids != 0
    inputs = {"input_ids": total_ids[:, :-1], "attention_mask": attn_mask[:, :-1]}

    if values[0] != {}:
        values = torch.stack(values)
    return inputs, total_label[:, 1:], nbest, values


def collate_fn_multiweak(batch):
    '''
    Only works for batch_size=1 for now.
    '''
    total_ids_list, total_label_list, nbest, values_list = zip(*batch)
    total_ids_list = total_ids_list[0]
    total_label_list = total_label_list[0]
    values_list = values_list[0]
    inputs_list = []
    for i in range(len(total_ids_list)):
        # total_ids_list[i].unsqueeze(0)
        # total_label_list[i].unsqueeze(0)
        total_ids_list[i] = torch.unsqueeze(total_ids_list[i], dim=0)
        total_label_list[i] = torch.unsqueeze(total_label_list[i], dim=0)
        total_label_list[i] = total_label_list[i][:, 1:]
        attention_mask = total_ids_list[i] != 0
        inputs_list.append({"input_ids": total_ids_list[i][:, :-1], "attention_mask": attention_mask[:, :-1]})
    return inputs_list, total_label_list, nbest, values_list