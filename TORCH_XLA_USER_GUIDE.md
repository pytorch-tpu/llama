## User Guide: Running Llama2 Inference on TPU v5e and v4

## Prerequisites



* Prepare the appropriate `params.json` for the target model using the following table
* If you need to run quantization, add `"quant": true` to your json file

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



## Commands to Run on TPU v5e


## Commands to Run on TPU v4


## Docker

Here is another approach where you can use a pre-built docker that has everything and then just use the following scripts to run the docker on any TPUs.


### Stack Versions



1. [git@github.com](mailto:git@github.com):pytorch-tpu/llama.git @ b89dd0f2351c42fef367670d9d2c5b65cd0ae932
2. PyTorch/XLA nightly @ 8.17.2023
3. PyTorch nightly @ 8.17.2023
4. Libtpu @ 0.1.dev20230809
5. Python @ 3.8.17


### Docker Image



* us-central1-docker.pkg.dev/jonbolin-test/jonbolin-test/jwtan-llama-inference:latest
* [Link](https://pantheon.corp.google.com/artifacts/docker/jonbolin-test/us-central1/jonbolin-test/jwtan-llama-inference/sha256:7e138a12616dbd3d429098514440670ba6a640d8ccb234924c938ba0bd21d35c?e=13802955&jsmode=o&mods=allow_workbench_image_override&project=jonbolin-test)
* Noted: if you ever have troubles on getting the docker images, let  know.


### Steps


#### 1. Copy all the following files


##### benchmark_inference.sh


##### env_inference.sh


##### host_script_inference.sh


#### 2. Modify the above files for your specific run configs.


#### 3. Run benchmark_inference.sh


#### 4. SSH to either of the workers and you should see the profiles under plugins/profile/ and then copy it to a gs bucket.


#### 5. Then follow  to upload the profile to xprof.


## Appendix


### 7B config


### 13B config


### 70B config
