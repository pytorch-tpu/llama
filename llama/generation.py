# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import time
from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer

import os
USE_CUDA = os.environ.get('USE_CUDA', False)

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
            self._generate_one_token_fn = torch.compile(self._generate_one_token_fn,
                                                    backend="torchxla_trace_once", fullgraph=True)

    def _generate_one_token(self, tokens, input_tokens, input_text_mask, cur_pos_tensor,
                            input_pos_tensor, output_pos_tensor, cache_kvs, temperature, top_p):
        logits, cache_kvs = self.model(input_tokens, input_pos_tensor, output_pos_tensor, cache_kvs)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        input_text_mask_tmp = input_text_mask.index_select(1, cur_pos_tensor).squeeze(dim=1)
        tokens_tmp = tokens.index_select(1, cur_pos_tensor).squeeze(dim=1)
        next_token = torch.where(
            input_text_mask_tmp, tokens_tmp, next_token
        )
        next_token = next_token.unsqueeze(1)
        tokens = tokens.index_copy(1, cur_pos_tensor, next_token)
        input_pos_tensor = input_pos_tensor[-1:] + 1
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
        prompt_tokens = [self.tokenizer.encode(x, bos=bos, eos=False) for x in prompts]

        total_len = params.max_seq_len

        tokens = torch.full((params.max_batch_size, total_len), self.tokenizer.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        tokens = tokens.to(device)
        input_text_mask = tokens != self.tokenizer.pad_id

        start_pos = 1
        cur_pos_tensor = torch.tensor(start_pos).to(device)
        input_pos_tensor = torch.arange(0, start_pos).to(device)
        output_pos_tensor = cur_pos_tensor - 1
        input_tokens = tokens.index_select(1, input_pos_tensor)
        cache_kvs = self.model.cache_kvs
        if device.type == "xla":
            xm.mark_step(wait=True)

        decoding_start_time = time.time()
        for _ in range(start_pos, total_len):
            tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs \
                = self._generate_one_token_fn(
                    tokens, input_tokens, input_text_mask, cur_pos_tensor,
                    input_pos_tensor, output_pos_tensor, cache_kvs, temperature, top_p
                )
            if device.type == "xla":
                xm.mark_step()
        self.model.cache_kvs = cache_kvs
        print(f"Decoded in {time.time() - decoding_start_time:.5f} seconds")

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            if i >= len(prompt_tokens):
                break
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
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
    mask = probs_sum - probs_sort > p
    probs_sort = torch.where(mask, 0.0, probs_sort)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
