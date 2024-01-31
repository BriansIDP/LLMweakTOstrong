import os
import re
import math
import pathlib
import random
from typing import Optional, Dict
from tqdm import tqdm
import json
from collections import defaultdict
import pickle

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

    def update_with_firstpass(self, labelset, threshold):
        for datapiece in self.main_data:
            datapiece[-2] = []
            for item in labelset[datapiece[0]]:
                if threshold < 0:
                    item[0] = item[0] + "<uncertainty: {:.2f}>".format(item[1])
                elif item[1] > threshold:
                    item[0] = item[0] + " <uncertain>"
                datapiece[-2].append(item[0])

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
            return total_ids, total_label, nbest_prompt
        else:
            return slurpid, prompt, nbest_prompt, label


def collate_fn_active(batch):
    slurp_ids, sequences, nbest_prompt, label = zip(*batch)
    return slurp_ids, sequences, nbest_prompt, label


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        slots,
        slotstr,
        prompts,
        knowledge="",
        model_max_length=2048,
        linearise=False,
        generating=False,
        maxKBsize=0,
        KBdrop=0,
        task="slot",
        LLM="vicuna",
        nbest=1,
    ):
        super(SupervisedDataset, self).__init__()
        self.linearise = linearise
        self.maxKBsize = maxKBsize
        self.nbest = nbest
        self.task = task
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.KBdrop = KBdrop
        if task == "slot":
            self.knowledge = self.get_knowledge_index(knowledge) if knowledge else {}
            self.data = self.process_json_data(json.load(open(data_path)))
            self.slots = slots
            self.slotstr = slotstr
        else:
            self.data = self.process_json_data_qa(json.load(open(data_path)))
        self.model_max_length = model_max_length
        self.generating = generating
        self.LLM = LLM

    def __len__(self):
        return len(self.data)

    def get_knowledge_index(self, knowledge):
        knowledge_index = {}
        with open(knowledge) as fin:
            data = json.load(fin)
            for key, values in data.items():
                for value in values:
                    knowledge_str = "{} is a type of {}".format(value, key)
                    if value not in knowledge_index:
                        knowledge_index[value] = []
                    knowledge_index[value].append(knowledge_str)
        return knowledge_index

    def process_json_data(self, data):
        sludata = []
        values = {
            "<slot>": [],
            "<value>": {},
        }
        for utterance in data["data"]:
            label = {}
            localvalues = {
                "<slot>": [],
                "<value>": {},
            }
            random.shuffle(utterance["entities"])
            for ent in utterance["entities"]:
                if ent["type"] in label:
                    label[ent["type"]] += " & " + ent["value"]
                else:
                    label[ent["type"]] = ent["value"]

            if label != {}:
                key = random.choice(list(label.keys()))
                keystring = json.dumps({key: label[key]})
            else:
                keystring = "{}"

            uttlist = []
            if "nbest" in utterance:
                for utt in utterance["nbest"]:
                    uttlist += utt.split()
                    uttlist.append("\n")
            else:
                uttlist += utterance["text"].split()

            for value, content in self.knowledge.items():
                inutt = False
                value = value.split()
                for i in range(len(uttlist)):
                    if value == uttlist[i:i+len(value)]:
                        inutt = True
                if inutt: # and random.random() > self.KBdrop:
                    for kitem in content:
                        value, slot = kitem.split(" is a type of ")
                        if slot not in localvalues["<slot>"]:
                            localvalues["<slot>"].append(slot)
                            localvalues["<value>"][slot] = [value]
                        else:
                            localvalues["<value>"][slot].append(value)

            labelstr = json.dumps(label)
            if "nbest" in utterance:
                utterance["nbest"].append(utterance["text"])
                sludata.append([utterance["nbest"], labelstr, localvalues, keystring, localvalues])
            else:
                sludata.append([utterance["text"], labelstr, localvalues, keystring, localvalues])

        return sludata

    def process_json_data_qa(self, data):
        qadata = []
        print("Start processing data")
        for i, datapiece in enumerate(tqdm(data)):
            keystrings = []
            content = datapiece["question"]
            label = random.choice(datapiece["answers"])["text"]
            keystrings.append(label)
            the_true_path = []
            for path in datapiece["truepath"]:
                if isinstance(path[0], list):
                    for partial in path:
                        partial[1] = " ".join(partial[1].split(".")[1:]).replace("_", " ")
                        partial[1] = "is {} of".format(partial[1])
                        if label in partial:
                            the_true_path = path
                else:
                    if label in path:
                        path[1] = " ".join(path[1].split(".")[1:]).replace("_", " ")
                        path[1] = "is {} of".format(path[1])
                        the_true_path = [path]
            truepath = " and ".join([" ".join(path) for path in the_true_path])
            keystrings.extend([" ".join(path) for path in the_true_path])

            format_str = self.prompts["direct"] # random.choice(self.prompts["format"])
            labelformat = format_str.format(**locals())
            valuepath = os.path.join("/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMknowledge/data/knowledge/webqsp_knowledge", datapiece["id"] + ".pkl")
            if os.path.exists(valuepath):
                with open(valuepath, "rb") as inp:
                    values = pickle.load(inp)
                setattr(values, "task", "qa")
                # LKI with noise
                truepaths = [truepath] if random.random() > self.KBdrop else []
                while len(truepaths) < self.maxKBsize:
                    KGpath = [random.choice(values.groupliteral["<triplet_1>"])]
                    if KGpath[-1] in values.groupliteral["<triplet_2>"]:
                        KGpath.append(random.choice(values.groupliteral["<triplet_2>"][KGpath[-1]]))
                    truepaths.append(" and ".join(KGpath))
                random.shuffle(truepaths)
                qadata.append((content, labelformat, values, truepath, truepaths))
                continue

            # values = {}
            values = {
                "<triplet_1>": [],
                "<triplet_2>": {},
                "<entity>": []
            }
            mapping = defaultdict(list)
            for triplet in datapiece["first_order"]:
                relation = " ".join(triplet[1].split(".")[1:]).replace("_", " ")
                relation = "is {} of".format(relation)
                triplet_str = " ".join([triplet[0], relation, triplet[2]])
                if len(triplet_str.split()) < 30 and len(triplet[0]) < 50 and len(triplet[2]) < 50:
                    if triplet_str not in values["<triplet_1>"]:
                        values["<triplet_1>"].append(triplet_str)

                    if triplet[0] not in mapping:
                        mapping[triplet[0]] = [triplet_str]
                    else:
                        mapping[triplet[0]].append(triplet_str)
                    if triplet[2] not in mapping:
                        mapping[triplet[2]] = [triplet_str]
                    else:
                        mapping[triplet[2]].append(triplet_str)
                    if triplet[0] not in values["<entity>"] and triplet[0] not in datapiece["entities"]:
                        values["<entity>"].append(triplet[0])
                    if triplet[2] not in values["<entity>"] and triplet[2] not in datapiece["entities"]:
                        values["<entity>"].append(triplet[2])

            for triplet in datapiece["second_order"]:
                relation = " ".join(triplet[1].split(".")[1:]).replace("_", " ")
                relation = "is {} of".format(relation)
                triplet_str = " ".join([triplet[0], relation, triplet[2]])
                leading_rels = []

                if len(triplet_str.split()) < 30 and len(triplet[0]) < 50 and len(triplet[2]) < 50:
                    if triplet[0] in mapping:
                        leading_rels.extend(mapping[triplet[0]])
                    if triplet[2] in mapping:
                        leading_rels.extend(mapping[triplet[2]])
                    for relation_obj in leading_rels:
                        if relation_obj not in values["<triplet_2>"]:
                            values["<triplet_2>"][relation_obj] = []
                        if triplet_str not in values["<triplet_2>"][relation_obj]:
                            values["<triplet_2>"][relation_obj].append(triplet_str)
                
                    if triplet[0] not in values["<entity>"] and triplet[0] not in datapiece["entities"]:
                        values["<entity>"].append(triplet[0])
                    if triplet[2] not in values["<entity>"] and triplet[2] not in datapiece["entities"]:
                        values["<entity>"].append(triplet[2])

            # values = LexFSA(self.tokenizer, task=self.task, groupliteral=values)
            # LKI with noise
            truepaths = [truepath] if random.random() > self.KBdrop else []
            # while len(truepaths) < self.maxKBsize:
            #     KGpath = [random.choice(values.groupliteral["<triplet_1>"])]
            #     if KGpath[-1] in values.groupliteral["<triplet_2>"]:
            #         KGpath.append(random.choice(values.groupliteral["<triplet_2>"][KGpath[-1]]))
            #     truepaths.append(" and ".join(KGpath))
            # random.shuffle(truepaths)

            # with open(valuepath, "wb") as fout:
            #     pickle.dump(values, fout, pickle.HIGHEST_PROTOCOL)
            qadata.append((content, labelformat, truepaths, truepath, truepaths))
        return qadata

    def get_split_mask(self, label_ids, keystrings):
        keyids = []
        for keystring in keystrings:
            keyid = self.tokenizer.encode(keystring)[1:]
            keyids.append(keyid)
        labelmask = label_ids * 0
        for keyid in keyids:
            for i in range(len(labelmask)):
                if keyid == label_ids[i:i+len(keyid)].tolist():
                    labelmask[i:i+len(keyid)] = 1
        return labelmask

    def preprocessing(self, sample):
        content, label, values, keystrings, truepaths = sample
        if isinstance(content, list):
            content = "\n".join(random.choices(content, k=self.nbest))
        knowledge = ""
        if self.task == "slot":
            knowledge_items = {}
            for key, value in truepaths["<value>"].items():
                vallist = [val for val in value if random.random() > self.KBdrop]
                if vallist != []:
                    knowledge_items[key] = vallist
            knowledge = json.dumps(knowledge_items)
        system = self.prompts["system"]
        taskdesc = self.prompts["task_description"].format(self.slotstr) if self.task == "slot" else ""
        query = self.prompts["query"] if self.nbest <= 1 else self.prompts["user2nbest"]

        knowledge_prompt = None
        if self.linearise and isinstance(knowledge, str):
            prompt = templates[self.LLM][self.task][1].format(**locals())
            knowledge_prompt = templates[self.LLM][self.task][2].format(**locals())
        else:
            prompt = templates[self.LLM][self.task][0].format(**locals())
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        input_size = len(prompt_inputs["input_ids"][0]) - 5
        label_ids = self.tokenizer(label + "</s>", return_tensors="pt")["input_ids"]
        label_ids = label_ids[0, 1:] if label_ids[0, 0] == 1 else label_ids[0]
        total_ids = torch.cat([prompt_inputs["input_ids"][0], label_ids], dim=-1)
        total_label = torch.cat([prompt_inputs["input_ids"][0] * 0 - 1, label_ids], dim=-1)

        knowledge_ids, knowledge_label = None, None
        if knowledge_prompt is not None and isinstance(keystrings, str):
            knowledge_prompt = self.tokenizer(knowledge_prompt, return_tensors="pt")
            knowledgelabel = self.tokenizer(keystrings + "</s>", return_tensors="pt")["input_ids"][0, 1:]
            knowledge_ids = torch.cat([knowledge_prompt["input_ids"][0], knowledgelabel], dim=-1)
            knowledge_label = torch.cat([knowledge_prompt["input_ids"][0] * 0 - 1, knowledgelabel], dim=-1)
        
        return total_ids, total_label, values, input_size, knowledge_ids, knowledge_label

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

def collate_fn(batch):
    total_ids, total_label, nbest  = zip(*batch)

    total_ids = pad_sequence(total_ids, batch_first=True, padding_value=1).to(device)
    total_label = pad_sequence(total_label, batch_first=True, padding_value=-1).to(device)
    attn_mask = total_ids != 0
    inputs = {"input_ids": total_ids[:, :-1], "attention_mask": attn_mask[:, :-1]}

    return inputs, total_label[:, 1:], nbest
