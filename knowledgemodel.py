import numpy as np
import six
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


class Hypo:
    # 每一个beam对应一个hypo
    def __init__(self):
        self.yseq = []              # sequence
        self.scores = []            # score for each token
        self.cumscore = 0.0         # sum of score
        self.normscore = 0.0        # sum of score / length
        self.entropy = []           # entropy for each token
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
        return outputs, labels

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

        if self.llm.config.model_type == "bloom":
            past_key_values = self.llm._convert_to_bloom_cache(past_key_values)

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

    def scoring(self, prompt, label):
        sequence = prompt + label
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            logits = self.llm(**inputs).logits
        length_x = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        # masked_logits = logits[0, length_x:, :]
        # masked_logprob = torch.log_softmax(masked_logits, dim=-1)
        # entropy = -(torch.exp(masked_logprob) * masked_logprob / len(masked_logprob)).sum()
        logits = logits.to(torch.float32)
        logprobs = torch.log_softmax(logits, dim=-1)
        # token_logprobs = torch.gather(logprobs, dim=2, index=inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
        token_logprobs = logprobs[0, :, inputs["input_ids"].squeeze()]
        token_logprobs = torch.diag(token_logprobs, diagonal=1)
        token_logprobs = token_logprobs[length_x-1:]
        score = token_logprobs.mean()
        return score