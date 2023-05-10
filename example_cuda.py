# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as xmp
import fire

from llama.xla_model_parallel import set_g_group
from example_xla import load


def setup_model_parallel(rank, world_size) -> Tuple[int, int]:
    # assuming model parallelism over the whole world size
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    set_g_group()

    # seed must be the same in all processes
    torch.manual_seed(1)


def main(
    rank: int,
    world_size: int,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    ckpt_dir: str = '',
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
):
    print(ckpt_dir)
    setup_model_parallel(rank, world_size)
    if rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, rank,
                     world_size, max_seq_len, max_batch_size,
                     torch.device("cuda", rank), dim, n_layers, n_heads)

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        # "Simply put, the theory of relativity states that ",
        # "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        #        """Tweet: "I hate it when my phone battery dies."
        #Sentiment: Negative
        ####
        #Tweet: "My day has been 👍"
        #Sentiment: Positive
        ####
        #Tweet: "This is the link to the article"
        #Sentiment: Neutral
        ####
        #Tweet: "This new music video was incredibile"
        #Sentiment:""",
        #        """Translate English to French:
        #
        #sea otter => loutre de mer
        #
        #peppermint => menthe poivrée
        #
        #plush girafe => girafe peluche
        #
        #cheese =>""",
    ]
    for _ in range(2):
        with torch.no_grad():
            results = generator.generate(prompts,
                                         256,
                                         torch.device("cuda", rank),
                                         temperature=temperature,
                                         top_p=top_p)

        for result in results:
            print(result)
            print("\n==================================\n")


def _fn(
    rank: int,
    world_size: int,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    ckpt_dir: str = '',
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
):
    main(rank, world_size, tokenizer_path, temperature, top_p, max_seq_len,
         max_batch_size, ckpt_dir, dim, n_layers, n_heads)


def mp_main(
    mp: bool,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    ckpt_dir: str = '',
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
):
    world_size = 1
    assert torch.cuda.is_available(), "cuda is not available"
    if mp:
        world_size = torch.cuda.device_count()
        print(f"Spawning {world_size} processes")
        xmp.spawn(_fn,
                  args=(world_size, tokenizer_path, temperature, top_p,
                        max_seq_len, max_batch_size, ckpt_dir, dim, n_layers,
                        n_heads),
                  nprocs=world_size,
                  join=True)
    else:
        main(0, world_size, tokenizer_path, temperature, top_p, max_seq_len,
             max_batch_size, ckpt_dir, dim, n_layers, n_heads)


if __name__ == "__main__":
    fire.Fire(mp_main)
