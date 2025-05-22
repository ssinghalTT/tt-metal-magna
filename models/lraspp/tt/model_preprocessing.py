# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.lraspp.reference.lraspp import (
    Conv2dNormActivation,
    InvertedResidual,
)
import torch.nn as nn

from models.experimental.lraspp.tt.seg_transforms import Compose, RandomResize, Normalize, DeNormalize
from models.experimental.lraspp.tt.segmentation import BinarySegmentationData
import numpy as np
from pdb import set_trace as bp

def get_fire_dataset_transform():
    train_transform = Compose(
        [
            RandomResize(max_size=224,min_size=224),
            Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ]
    )
    return train_transform
    
def get_fire_dataset_inverse_transform():
    transform = DeNormalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    return transform

def create_lraspp_input_tensors(batch=1, input_channels=3, input_height=224, input_width=224):
    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    # ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    # ttnn_input_tensor = ttnn_input_tensor.reshape(
    #    1,
    #    1,
    #    ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
    #    ttnn_input_tensor.shape[3],
    # )
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return torch_input_tensor, ttnn_input_tensor

def create_lraspp_input_tensors_fire(batch=1, input_channels=3, input_height=224, input_width=224):
    train_transform = get_fire_dataset_transform()
    # bp()

    dataset_path    = ["./models/experimental/lraspp/tests/images/Image/Fire"]
    mask_path       = ["./models/experimental/lraspp/tests/images/Segmentation_Mask/Fire"]
    sim_data = BinarySegmentationData(train_dir=dataset_path,
                                      mask_dir=mask_path,
                                      train_transform=train_transform)
    _, val_set = sim_data.split()
    indices = np.random.permutation(len(val_set))[:batch]

    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    torch_output_tensor = torch.randn(batch, 1, input_height, input_width)
    for i, idx in enumerate(indices):
        img, mask = val_set[idx]
        torch_input_tensor[i] = img
        torch_output_tensor[i] = mask

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output_tensor = ttnn.from_torch(torch_output_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return torch_input_tensor, ttnn_input_tensor, torch_output_tensor, ttnn_output_tensor

def fold_batch_norm2d_into_conv2d(conv, bn, bfloat8_b=True):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into Conv2d")
    weight = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))
    bias = torch.reshape(bias, (1, 1, 1, -1))
    if bfloat8_b:
        weight = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = ttnn.from_torch(bias, dtype=ttnn.float32)
    else:
        weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
    return weight, bias


def preprocess_conv_parameters(conv, bias=True, bfloat8_b=True):
    weight = conv.weight
    if bias:
        bias = conv.bias
        bias = torch.reshape(bias, (1, 1, 1, -1))
        if bfloat8_b:
            bias = ttnn.from_torch(bias, dtype=ttnn.float32)
        else:
            bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
    else:
        bias = None
    if bfloat8_b:
        weight = ttnn.from_torch(weight, dtype=ttnn.float32)
    else:
        weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16)

    return weight, bias


def create_lraspp_model_parameters(model, device):
    model_parameters = {}
    conv_bn_counter = 0
    counter = 0

    for name, module in model.named_modules():
        if isinstance(module, InvertedResidual):
            for idx, submodule in enumerate(module.conv):
                if isinstance(submodule, nn.Conv2d):
                    bn = (
                        module.conv[idx + 1]
                        if idx + 1 < len(module.conv) and isinstance(module.conv[idx + 1], nn.BatchNorm2d)
                        else None
                    )
                    if bn:
                        weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv2d(submodule, bn)
                        model_parameters[f"conv_{counter}_weight"] = weight_ttnn
                        model_parameters[f"conv_{counter}_bias"] = bias_ttnn
                        counter += 1

        elif isinstance(module, Conv2dNormActivation):
            if len(module) == 3 and isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.BatchNorm2d):
                conv = module[0]
                bn = module[1]
                weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv2d(conv, bn)
                model_parameters[f"fused_conv_{conv_bn_counter}_weight"] = weight_ttnn
                model_parameters[f"fused_conv_{conv_bn_counter}_bias"] = bias_ttnn
                conv_bn_counter += 1

    model_parameters[53] = fold_batch_norm2d_into_conv2d(model.__getattr__(f"c{53}"), model.__getattr__(f"b{53}"))
    model_parameters[54] = preprocess_conv_parameters(model.__getattr__(f"c{54}"), bias=False)

    model_parameters["low_classifier"] = preprocess_conv_parameters(model.__getattr__("low_classifier"))
    model_parameters["high_classifier"] = preprocess_conv_parameters(model.__getattr__("high_classifier"))

    return model_parameters
