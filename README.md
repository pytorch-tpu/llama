# LLaMA

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Setup

Prepare TPU cluster environment variables:
```
export TPU_NAME=<your tpu vm name>
export PROJECT=<your gcloud project name>
export ZONE=<your tpu vm zone>
```

Install nightly torch and torch-xla packages:
```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all --command='
sudo pip3 uninstall torch torch_xla libtpu-nightly torchvision -y
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly+20230422-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly+20230422-cp38-cp38-linux_x86_64.whl
pip3 install torch-xla[tpuvm]
'
```

Download repo and install dependencies on the TPU VM:
```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all --command='
git clone --branch stable https://github.com/pytorch-tpu/llama.git
cd llama
pip3 install -r requirements.txt
pip3 install -e .
'
```

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

## Inference

The provided `example_xla.py` can be run on a TPU VM with `gcloud compute tpus tpu-vm ssh` and will output completions for one pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all --command='
export PJRT_DEVICE=TPU
export TOKENIZER_PATH=$TARGET_FOLDER/tokenizer.model
export CKPT_DIR=$TARGET_FOLDER/model_size

cd llama
python3 example_xla.py --tokenizer_path $TOKENIZER_PATH --ckpt_dir $CKPT_DIR --max_seq_len 256 --max_batch_size 1 --temperature 0.8 --mp True
'
```

If you don't have the downloaded LLaMA model files, you can try the script with the provided T5 tokenizer (note that without a model checkpoint, the output would not make sense):
```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all --command='
export PJRT_DEVICE=TPU
export TOKENIZER_PATH=$HOME/llama/t5_tokenizer/spiece.model

cd llama
python3 example_xla.py --tokenizer_path $TOKENIZER_PATH --max_seq_len 256 --max_batch_size 1 --temperature 0.8 --dim 4096 --n_heads 32 --n_layers 32 --mp True
'
```

If the downloaded checkpoint has a different model parallelism world size than the targeted TPU VM world size, script `reshard_checkpoints.py` can be used to re-shard the model checkpoint to more pieces. For example, to reshard a 13B LLaMA model checkpoint to run on a V4-16 TPU slice, which has 8 devices:
```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all --command='
export PJRT_DEVICE=TPU
export TOKENIZER_PATH=$TARGET_FOLDER/tokenizer.model
export CKPT_DIR=$TARGET_FOLDER/model_size
export NEW_CKPT_DIR=$TARGET_FOLDER/model_size/resharded

cd llama
python3 reshard_checkpoints.py --original_mp 2 --target_mp 8 --ckpt_dir $CKPT_DIR --tokenizer_path $TOKENIZER_PATH --output_dir $NEW_CKPT_DIR
'
```

Different models have different original_mp values:
|  Model | original_mp |
|--------|-------------|
| 7B     | 1           |
| 13B    | 2           |
| 33B    | 4           |
| 65B    | 8           |

### XLA GPU

`example_xla.py` can also be ran on GPUs with XLA:GPU.
```
PJRT_DEVICE=GPU GPU_NUM_DEVICES=4 python3 example_xla.py --tokenizer_path $TOKENIZER_PATH --max_seq_len 256 --max_batch_size 1 --temperature 0.8 --dim 4096 --n_heads 32 --n_layers 32 --mp True
```

## CUDA

`example_cuda.py` is provided to produce CUDA (using Inductor by default) results as the same format as `example_xla.py` such that one can easily compare
results among XLA:TPU, XLA:GPU, CUDA eager, CUDA Inductor.

Here is how you can use it:
```
USE_CUDA=1 python3 example_cuda.py --tokenizer_path $TOKENIZER_PATH --max_seq_len 256 --max_batch_size 1 --temperature 0.8 --dim 4096 --n_heads 32 --n_layers 32 --mp True
```

## FAQ

- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.
