from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.ao.quantization.fx._decomposed
from typing import Optional

EPS = torch.finfo(torch.float32).eps

@dataclass
class TensorQConfig:
    dtype: torch.dtype = torch.int8
    axis: int = -1
    quant_min: int = -128
    quant_max: int = 127
    symmetric_quant: bool = True


def _get_dtype_min_max(dtype: torch.dtype):
    if dtype == torch.int8:
        return -128, 127
    elif dtype == torch.uint8:
        return 0, 127
    else:
        assert False

def _find_per_channel_min_max(x: torch.Tensor, axis: int):
    x_dim = x.size()
    new_axis_list = [i for i in range(len(x_dim))]
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(new_axis_list)
    y = torch.flatten(y, start_dim=1)
    return torch.aminmax(y, dim=1)

def _find_qparams(x: torch.Tensor, qconfig : TensorQConfig):
    # Only support per-channel symmetric quant to int8 now
    axis = qconfig.axis
    dtype = qconfig.dtype
    symmetric_quant = qconfig.symmetric_quant
    quant_min = qconfig.quant_min
    quant_max = qconfig.quant_max
    assert axis >= 0 and axis < len(x.shape)
    assert dtype == torch.int8
    min_val, max_val = _find_per_channel_min_max(x, axis)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    scale = torch.ones(min_val_neg.size(), dtype=torch.float32)
    if symmetric_quant:
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        eps = torch.zeros_like(scale).fill_(EPS)
        scale = torch.max(scale, eps)
        return scale, None
    else:
        assert symmetric_quant

def _quantize_to_dtype(x: torch.Tensor, qconfig: TensorQConfig,
                       scale: torch.Tensor,
                       zero_point: Optional[torch.Tensor] = None):
    if zero_point is None:
        zero_point = torch.zeros_like(scale)
    return torch.ops.quantized_decomposed.quantize_per_channel(
        x, scale, zero_point, qconfig.axis, qconfig.quant_min,
        qconfig.quant_max, qconfig.dtype
    )

def quantize_tensor(x: torch.Tensor, qconfig : TensorQConfig):
    scale, zp = _find_qparams(x, qconfig)
    x_int = _quantize_to_dtype(x, qconfig, scale, zp)
    return x_int, scale, zp
