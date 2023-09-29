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
