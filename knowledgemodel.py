import argparse
import logging
import math
import os
from time import time
from copy import deepcopy
import random
import json
from copy import deepcopy

import numpy as np
import six
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel


default_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    # target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
)


class Hypo:
    def __init__(self):
        self.yseq = []
        self.scores = []
        self.cumscore = 0.0
        self.normscore = 0.0
        self.entropy = []
        self.treetrack = []
        self.completed = []
        self.completed_state = []

class CrossAttention(torch.nn.Module):
    def __init__(self, input_dim, query_dim, attention_dim, n_query):
        super(CrossAttention, self).__init__()
        # self.qproj = torch.nn.Linear(query_dim, attention_dim)
        # self.kproj = torch.nn.Linear(input_dim, attention_dim)
        # self.vproj = torch.nn.Linear(input_dim, attention_dim)
        self.attention_drop = torch.nn.Dropout(0.1)
        self.nhead = n_query

    def forward(self, inputs, query, weightmask):
        Q = query # self.attention_drop(self.qproj(query))
        K = inputs # self.attention_drop(self.kproj(inputs))
        V = inputs # self.attention_drop(self.kproj(inputs))
        attention_weights = torch.einsum("btj,ij->bti", Q, K) # / math.sqrt(Q.size(-1))
        attention_weights = attention_weights + weightmask * -1e9
        attention_weights = torch.softmax(attention_weights, dim=-1)
        output = torch.einsum("bti,ij->btj", attention_weights, V)
        return attention_weights, output


class KnowledgeLLM(torch.nn.Module):
    def __init__(
        self,
        llm,
        tokenizer,
        ontology=None,
        knowledge_dim=0,
        KBsize=0,
        KBdrop=0,
        maxKB=0,
        useptr=False,
        nquery=1,
        task="slot",
        peft_config="",
    ):
        super(KnowledgeLLM, self).__init__()
        self.llm = llm
        self.use_lora = True
        self.knowledge_dim = knowledge_dim
        self.maxKB = maxKB
        self.KBsize = KBsize
        self.KBdrop = KBdrop
        self.KBindices = [i for i in range(self.KBsize)]
        self.useptr = useptr
        self.attndim = 1024
        self.tokenizer = tokenizer
        self.ontology = ontology
        self.pointer = 0.0
        self.task = task

    def forward(self, inputs, labels, knowledge=None, values=None, knowledge_prompt=None, knowledge_label=None):
        outputs = self.llm(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        knowledgeoutput = None
        if self.useptr and knowledgeoutput is None:
            attention_masks = self.process_label_copy(labels, values)
            outputs.logits = self.calc_pointer(outputs, attention_masks)
        return outputs, labels, knowledgeoutput

    def calc_pointer(self, outputs, attention_masks, kflag=False):
        hidden_state = outputs.hidden_states[-1]
        if self.use_lora:
            atten_weights, ptr_output = self.ptrmodel(self.llm.model.lm_head.weight, hidden_state, attention_masks)
        else:
            atten_weights, ptr_output = self.ptrmodel(self.llm.lm_head.weight, hidden_state, attention_masks)
        pointer_prob = torch.sigmoid(self.ptr_proj(torch.cat([hidden_state, ptr_output], dim=-1)))
        # print(pointer_prob[0, -1].item())
        llm_prob = torch.softmax(outputs.logits, dim=-1)
        final_prob = pointer_prob * atten_weights + (1 - pointer_prob) * llm_prob
        return torch.log(final_prob)

    def process_label_copy(self, labels, values, kflag=False):
        attention_masks = []
        for i, label in enumerate(labels):
            treetrack = []
            completed = []
            completed_state = []
            attention_mask = []
            treetrack, nexttokens, completed, completed_state = values[i].get_next_state(
                0, treetrack, completed, completed_state, kflag=kflag)
            for label_idx in label:
                label_idx = label_idx.item()
                if label_idx != -1:
                    step_mask = label.new_ones(self.llm.config.vocab_size)
                    step_mask[nexttokens] = 0
                    treetrack, nexttokens, completed, completed_state = values[i].get_next_state(
                        label_idx, treetrack, completed, completed_state, kflag=kflag)
                else:
                    step_mask = label.new_zeros(self.llm.config.vocab_size)
                attention_mask.append(step_mask)
            attention_masks.append(torch.stack(attention_mask, dim=0))
        return torch.stack(attention_masks, dim=0)

    def get_embedding(self, input_ids):
        if self.use_lora:
            input_embs = self.llm.model.model.embed_tokens(input_ids)
        else:
            input_embs = self.llm.model.embed_tokens(input_ids)
        return input_embs

    def generate(self, inputs, max_new_tokens=128, beamsize=1):
        output_sequences = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=beamsize,
            num_return_sequences=beamsize,
            return_dict_in_generate=True,
            output_scores=True,
        )
        return output_sequences

    def decode_one_step(self, input_ids, n_adapters, past_key_values=None):
        new_past_key_values = None
        output = self.llm(
            input_ids,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )
        new_past_key_values = output.past_key_values
        masked_logprob = torch.log_softmax(output.logits[:, -1], dim=-1)
        return masked_logprob, new_past_key_values

    def generate_beam(
            self,
            input_ids,
            max_new_tokens=256,
            stopping_criteria=None,
            knowledge=None,
            values=None,
            beamsize=1,
            kflag=False,
            do_sample=False,
            n_adapters=1,
        ):
        # input_embs = self.get_embedding(input_ids)
        hyps = [Hypo() for _ in range(beamsize)]
        treetrack = []
        completed = []
        nexttokens = []
        completed_state = []
        maskbase = input_ids.new_ones(self.llm.config.vocab_size)

        decoded = []
        masked_logprob, past_key_values = self.decode_one_step(input_ids, n_adapters)

        next_token = masked_logprob.topk(k=beamsize)
        keepbeam = next_token[1][0]
        scores = next_token[0][0]

        for i, hyp in enumerate(hyps):
            hyp.yseq.append(keepbeam[i].item())
            hyp.cumscore += scores[i].item()
            hyp.scores.append(scores[i].item())
            hyp.entropy.append(-(torch.exp(masked_logprob) * masked_logprob)[0].sum())
            if self.useptr and values is not None:
                hyp.treetrack, nexttokens, hyp.completed, hyp.completed_state = values.get_next_state(
                    keepbeam[i].item(), treetrack, completed[:], completed_state[:], kflag=kflag)
            else:
                hyp.treetrack, hyp.completed, hyp.completed_state = treetrack, completed, completed_state

        finished_beam = []
        keepbeam = keepbeam.unsqueeze(-1)
        scores = scores.unsqueeze(-1)
        if n_adapters > 1:
            for p in range(n_adapters):
                past_key_values[p] = [[item.repeat(beamsize, 1, 1, 1) for item in items] for items in past_key_values[p]]
        else:
            past_key_values = [[item.repeat(beamsize, 1, 1, 1) for item in items] for items in past_key_values]

        while keepbeam.size(-1) < max_new_tokens and len(finished_beam) < beamsize:
            # input_embs = self.get_embedding(keepbeam)
            masked_logprob, _ = self.decode_one_step(
                keepbeam,
                n_adapters,
                past_key_values=past_key_values,
            )
            conditional_entropy = ((-torch.exp(masked_logprob) * masked_logprob).sum(dim=1) * torch.softmax(scores.squeeze(-1), dim=-1)).sum()
            entropy = (-torch.exp(masked_logprob) * masked_logprob).sum(dim=1)
            next_token = masked_logprob.topk(k=beamsize, dim=-1)
            expanded_next = next_token[1].view(-1).unsqueeze(-1)
            expanded_score = next_token[0].view(-1)
            expanded_beam = torch.cat([keepbeam.unsqueeze(1).repeat(1, beamsize, 1).view(keepbeam.size(0)*beamsize, -1), expanded_next], dim=-1)
            expanded_beam_score = scores.repeat(1, beamsize).view(-1) + expanded_score
            selected = torch.topk(expanded_beam_score, k=beamsize*2 if beamsize > 2 else beamsize)
            selected_indices = selected[1]
            new_hyps = []
            new_index = []
            for index in selected_indices:
                index = index.item()
                orig_beam_index = index // beamsize
                newhyp = Hypo()
                newhyp.yseq = hyps[orig_beam_index].yseq[:] + [expanded_next[index].item()]
                newhyp.cumscore = hyps[orig_beam_index].cumscore + expanded_score[index]
                newhyp.scores = hyps[orig_beam_index].scores[:] + [expanded_score[index].item()]
                newhyp.entropy = hyps[orig_beam_index].entropy + [entropy[orig_beam_index]]
                newhyp.treetrack = hyps[orig_beam_index].treetrack
                newhyp.completed = hyps[orig_beam_index].completed[:]
                newhyp.completed_state = hyps[orig_beam_index].completed_state[:]
                if expanded_beam[index][-1] == 2 or self.tokenizer.decode(expanded_beam[index]).endswith("</s>"):
                    finished_beam.append(newhyp)
                else:
                    # Controller search
                    new_hyps.append(newhyp)
                    new_index.append(index)
                if len(new_hyps) == beamsize:
                    break
            if new_hyps == []:
                break
            keepbeam = expanded_beam[new_index]
            scores = expanded_beam_score[new_index]
            hyps = new_hyps
        if len(finished_beam) == 0:
            finished_beam.append(hyps[0])
        for hyp in finished_beam:
            hyp.normscore = hyp.cumscore / len(hyp.yseq)
        sorted_hyps = sorted(finished_beam, key=lambda finished_beam: finished_beam.normscore, reverse=True)
        return sorted_hyps
