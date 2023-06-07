# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import time
from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer

import os

USE_CUDA = os.environ.get('USE_CUDA', False)

# Some how xla init will slow down the CUDA speed.
if not USE_CUDA:
    import torch_xla.core.xla_model as xm


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._generate_one_token_fn = self._generate_one_token
        if USE_CUDA:
            # Inductor errors out when compiles _generate_one_token_fn.
            # TODO(alanwaketan): figure out why.
            self.model = torch.compile(self.model, fullgraph=True)
        else:
            self._generate_one_token_fn = torch.compile(
                self._generate_one_token_fn,
                backend="torchxla_trace_once",
                fullgraph=True)

    def _generate_one_token(self, tokens, input_tokens, input_text_mask,
                            cur_pos_tensor, input_pos_tensor,
                            output_pos_tensor, cache_kvs, temperature_tensor,
                            top_p_tensor, with_temp):
        logits, cache_kvs = self.model(input_tokens, input_pos_tensor,
                                       output_pos_tensor, cache_kvs)
        if with_temp:
            probs = torch.softmax(logits / temperature_tensor, dim=-1)
            next_token = sample_top_p(probs, top_p_tensor)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        input_text_mask_tmp = input_text_mask.index_select(
            1, cur_pos_tensor).squeeze(dim=1)
        tokens_tmp = tokens.index_select(1, cur_pos_tensor).squeeze(dim=1)
        next_token = torch.where(input_text_mask_tmp, tokens_tmp, next_token)
        next_token = next_token.unsqueeze(1)
        tokens = tokens.index_copy(1, cur_pos_tensor, next_token)
        # prepare for the next iteration
        input_pos_tensor = cur_pos_tensor.unsqueeze(0)
        cur_pos_tensor = cur_pos_tensor + 1
        output_pos_tensor = cur_pos_tensor - 1
        input_tokens = tokens.index_select(1, input_pos_tensor)

        return tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        device: torch.device,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        start_time = time.time()

        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        bos = not USE_CUDA  # The supplied t5 tokenizer doesn't have bos. CUDA will error out but xla won't.
        prompt_tokens = [
            self.tokenizer.encode(x, bos=bos, eos=False) for x in prompts
        ]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        assert min_prompt_size >= 1 and max_prompt_size < params.max_seq_len

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((params.max_batch_size, params.max_seq_len),
                            self.tokenizer.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t).long()
        tokens = tokens.to(device)
        input_text_mask = tokens != self.tokenizer.pad_id

        # Passing tensors instead of floats into self._generate_one_token_fn,
        # so that different values would not trigger compilations of new graphs
        temperature_tensor = torch.tensor(float(temperature)).to(device)
        top_p_tensor = torch.tensor(float(top_p)).to(device)
        with_temp = temperature > 0

        cache_kvs = self.model.cache_kvs
        if device.type == "xla":
            xm.mark_step()

        decoding_start_time = time.time()
        prev_pos = 0
        buckets = [128, 256, 384, 512]
        assert params.max_seq_len % buckets[-1] == 0
        while prev_pos < min_prompt_size:
            remaining = min_prompt_size - prev_pos
            section_len = 0
            for bucket in buckets:
                if bucket >= remaining:
                    section_len = bucket
                    break
            if section_len == 0:
                section_len = buckets[-1]

            cur_pos = min(min_prompt_size, prev_pos + section_len)
            print(f"Processing prompt pos [{prev_pos}, {prev_pos + section_len}), section length {section_len}, cur_pos {cur_pos}")
            cur_pos_tensor = torch.tensor(cur_pos).to(device)
            input_pos_tensor = torch.arange(prev_pos, prev_pos + section_len).to(device)
            output_pos_tensor = cur_pos_tensor - 1
            input_tokens = tokens.index_select(1, input_pos_tensor)
            if device.type == "xla":
                xm.mark_step()

            tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs \
                = self._generate_one_token_fn(
                    tokens, input_tokens, input_text_mask, cur_pos_tensor,
                    input_pos_tensor, output_pos_tensor, cache_kvs,
                    temperature_tensor, top_p_tensor, with_temp
                )
            if device.type == "xla":
                xm.mark_step()

            prev_pos = cur_pos

        assert cur_pos_tensor.item() == prev_pos + 1 and prev_pos == min_prompt_size
        for _ in range(prev_pos + 1, total_len):
            tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs \
                = self._generate_one_token_fn(
                    tokens, input_tokens, input_text_mask, cur_pos_tensor,
                    input_pos_tensor, output_pos_tensor, cache_kvs,
                    temperature_tensor, top_p_tensor, with_temp
                )
            if device.type == "xla":
                xm.mark_step()
        self.model.cache_kvs = cache_kvs
        print(f"Processed prompts with {min_prompt_size} to {max_prompt_size} tokens, and generated {total_len - max_prompt_size} tokens")
        print(f"Totally decoded {total_len - 1} tokens in {time.time() - decoding_start_time:.5f} seconds")

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            if i >= len(prompt_tokens):
                break
            # cut to max gen len
            t = t[:len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[:t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            try:
                sentence = self.tokenizer.decode(t)
            except IndexError:
                sentence = self.tokenizer.decode(t[1:])
            decoded.append(sentence)
        print(f"Completed in {time.time() - start_time:.5f} seconds")
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort = torch.where(mask, 0.0, probs_sort)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
