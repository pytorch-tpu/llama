# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import os
import torch

from llama import Llama

USE_CUDA = os.environ.get('USE_CUDA', False)

# Some how xla init will slow down the CUDA speed.
if USE_CUDA:
    import torch.multiprocessing as xmp
else:
    import torch_xla.debug.profiler as xp
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_model as xm

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    dynamo: str = "openxla_eval",
    repeat: int = 2,
):
    if not USE_CUDA:
        server = xp.start_server(9012)
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        dynamo=dynamo,
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
#        "Simply put, the theory of relativity states that ",
#        """A brief message congratulating the team on the launch:
#
#        Hi everyone,
#
#        I just """,
#        # Few shot prompt (providing a few examples before asking model to complete more);
#        """Translate English to French:
#
#        sea otter => loutre de mer
#        peppermint => menthe poivrée
#        plush girafe => girafe peluche
#        cheese =>""",
    ]
    for i in range(repeat):
        # Automatically takes profiles, let's skip the cold run and only capture warm runs.
        if i == 1 and not USE_CUDA and xm.is_master_ordinal():
            import tempfile
            from threading import Thread
            profile_logdir = os.environ.get('PROFILE_LOGDIR', tempfile.mkdtemp())
            profile_duration = int(os.environ.get('PROFILE_DURATION_MS', 20000))
            trace = lambda: xp.trace('127.0.0.1:9012', profile_logdir, profile_duration)
            Thread(target=trace).start()

        with torch.no_grad():
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt, result in zip(prompts, results):
                print(prompt)
                print(f"> {result['generation']}")
                print("\n==================================\n")


def _fn(
    idx,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    dynamo: str = "openxla_eval",
    repeat: int = 2,
):
    if USE_CUDA:
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
        os.environ['RANK'] = str(idx)
        os.environ['LOCAL_RANK'] = str(idx)
    main(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_gen_len, max_batch_size, dynamo, repeat)


def mp_main(
    mp: bool,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    dynamo: str = "openxla_eval",
    repeat: int = 2,
):
    # Sanity-check the combination of USE_CUDA envvar and --dynamo flag.
    if USE_CUDA:
        # Eager mode on GPU or Inductor.
        assert not dynamo or dynamo == "inductor"
    else:
        # Eager CPU / OpenXLA+Lazytensor (if PJRT_DEVICE is set), or OpenXLA.
        assert not dynamo or dynamo == "openxla" or dynamo == "openxla_eval"

    if mp:
        if USE_CUDA:
            kwargs = {"nprocs": torch.cuda.device_count(),
                      "join": True}
        else:
            kwargs = {}
        xmp.spawn(_fn,
                  args=(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_gen_len, max_batch_size, dynamo, repeat), **kwargs)
    else:
        main(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_gen_len, max_batch_size, dynamo, repeat)


if __name__ == "__main__":
    fire.Fire(mp_main)
