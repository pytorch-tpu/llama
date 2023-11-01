## User Guide: Running Llama2 Inference on TPU v5e and v4

## Prerequisites



* Prepare the appropriate `params.json` (e.g. `params70b.json`) for the target model using the following table
* If you need to run quantization, add `"quant": true` to your json file
* Use the instructions below to make your `params.json` available to the model

<table>
  <tr>
   <td>
7B
   </td>
   <td><code>{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}</code>
   </td>
  </tr>
  <tr>
   <td>13B
   </td>
   <td><code>{"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": -1}</code>
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td><code>{"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": -1}</code>
   </td>
  </tr>
</table>

* In this setup, we offer t5_tokenizer as an exmaple. Feel free to replace it
  with other tokenizers as appropriate.

## Commands to Run Llama2 using XLA:TPU (TPU v5e)
```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker=all --command="
sudo apt-get update
sudo apt-get -y install libomp5
sudo pip3 install mkl mkl-include
sudo pip3 install tf-nightly tb-nightly tbp-nightly
sudo ln -s /usr/local/lib/libmkl_intel_lp64.so.2 /usr/local/lib/libmkl_intel_lp64.so.1
sudo ln -s /usr/local/lib/libmkl_intel_thread.so.2 /usr/local/lib/libmkl_intel_thread.so.1
sudo ln -s /usr/local/lib/libmkl_core.so.2 /usr/local/lib/libmkl_core.so.1
sudo git clone --branch llama2-google-next-inference https://github.com/pytorch-tpu/llama.git
sudo chmod -R 777 llama
sudo pip3 uninstall torch torch_xla libtpu-nightly torchvision -y
pip3 uninstall torch torch_xla libtpu-nightly torchvision -y
pip3 install torchvision --user 
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly-cp310-cp310-linux_x86_64.whl --user 
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl --user
pip3 install torch-xla[tpuvm]
sudo apt-get install libopenblas-dev

gcloud compute tpus tpu-vm scp params_70b.json ${TPU_NAME}:params.json --zone ${ZONE} --project ${PROJECT_ID} --worker=all

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker=all --command="
sudo chmod -R 777 llama
cd llama/
pip3 install -r requirements.txt
pip3 install -e ."

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker=all --command="cd $HOME/llama && 
PJRT_DEVICE=TPU XLA_FLAGS=--xla_dump_to=/tmp/dir_name PROFILE_LOGDIR=/tmp/home/ python3 example_text_completion.py --ckpt_dir . --tokenizer_path $HOME/llama/t5_tokenizer/spiece.model --max_seq_len 2048 --max_gen_len 1000 --max_batch_size 2 --mp True --dynamo True"
```

## Commands to Run Llama2 using XLA:TPU (TPU v4)

```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker=all --command="
sudo pip3 uninstall torch torch_xla libtpu-nightly torchvision -y
pip3 uninstall torch torch_xla libtpu-nightly torchvision -y
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
pip3 install torch-xla[tpuvm]
sudo git clone --branch llama2-google-next-inference https://github.com/pytorch-tpu/llama.git

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker=all --command="
sudo apt update
sudo apt-get install libopenblas-dev"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker=all --command="
sudo chmod -R 777 llama
cd llama/
pip3 install -r requirements.txt
pip3 install -e ."

gcloud compute tpus tpu-vm scp params_70b.json ${TPU_NAME}:params.json --zone ${ZONE} --project ${PROJECT_ID} --worker=all

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker=all --command="cd $HOME/llama && 
PJRT_DEVICE=TPU XLA_FLAGS=--xla_dump_to=/tmp/dir_name PROFILE_LOGDIR=/tmp/home/ python3.8 example_text_completion.py --ckpt_dir . --tokenizer_path $HOME/llama/t5_tokenizer/spiece.model --max_seq_len 2048 --max_gen_len 1000 --max_batch_size 2 --mp True --dynamo True"
```
## Commands to Run Llama2 using XLA:GPU (e.g. L4 or H100)


