# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from ttnn.model_preprocessing import ParameterDict, fold_batch_norm2d_into_conv2d
from torch import nn
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
import math
import torch.nn.functional as F
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from ttnn.model_preprocessing import ParameterDict


def input_preprocessing(x, N, C, H, W):
    x = ttnn.to_torch(x)
    x = torch.permute(x, (0, 3, 1, 2))
    x = x.reshape(N, C, H, W)
    return x


class Conv_mnist_like_model_Conv2D:
    def __init__(
        self,
        conv,
        bn=None,
        device=None,
        cache={},
        activation="relu",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.device = device
        self.batch_size = conv.batch_size
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = True
        self.cache = cache
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            math_fidelity=ttnn.MathFidelity.LoFi,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
        )
        config_override = conv.conv_blocking_and_parallelization_config_override
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]
        if bn is not None:
            weight, bias = fold_batch_norm2d_into_conv2d(conv.module, bn.module)
        else:
            weight, bias = conv.module.weight, conv.module.bias
        weight = weight
        if bias is not None:
            bias = torch.reshape(bias, (1, 1, 1, -1))
            self.bias = ttnn.from_torch(bias, dtype=ttnn.float32)
        else:
            self.bias = None
        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32)

    def __call__(self, x):
        x, output_height, output_width, self.weight, self.bias = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            batch_size=self.batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            conv_op_cache=self.cache,
            groups=self.groups,
        )
        return x


class Conv_mnist_like_model:
    def __init__(self, device, parameters: ParameterDict):
        self.device = device
        self.parameters = parameters
        self.conv1 = Conv_mnist_like_model_Conv2D(parameters.conv1, device=device)
        self.conv2 = Conv_mnist_like_model_Conv2D(parameters.conv2, device=device)
        self.conv3 = Conv_mnist_like_model_Conv2D(parameters.conv3, device=device)
        self.conv4 = Conv_mnist_like_model_Conv2D(parameters.conv4, device=device)
        self.fc1_weights = parameters.fc1.module.weight
        self.fc2_weights = parameters.fc2.module.weight
        self.fc1_bias = parameters.fc1.module.bias
        self.fc2_bias = parameters.fc2.module.bias
        self.pool3 = nn.MaxPool2d(3, 3)

    def __call__(self, x):
        x = self.conv1(x)
        x = input_preprocessing(
            x,
            self.parameters.pool3.batch_size,
            self.parameters.conv1.out_channels,
            self.parameters.pool3.input_height,
            self.parameters.pool3.input_width,
        )
        x = self.pool3(x)
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        x = self.conv2(x)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.parameters.pool2.batch_size,
            input_h=self.parameters.pool2.input_height,
            input_w=self.parameters.pool2.input_width,
            channels=self.parameters.conv2.out_channels,
            kernel_size=[self.parameters.pool2.kernel_size, self.parameters.pool2.kernel_size],
            stride=[self.parameters.pool2.stride, self.parameters.pool2.stride],
            padding=[self.parameters.pool2.padding, self.parameters.pool2.padding],
            dilation=[self.parameters.pool2.dilation, self.parameters.pool2.dilation],
        )
        x = self.conv3(x)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.parameters.pool4.batch_size,
            input_h=self.parameters.pool4.input_height,
            input_w=self.parameters.pool4.input_width,
            channels=self.parameters.conv3.out_channels,
            kernel_size=[self.parameters.pool4.kernel_size, self.parameters.pool4.kernel_size],
            stride=[self.parameters.pool4.stride, self.parameters.pool4.stride],
            padding=[self.parameters.pool4.padding, self.parameters.pool4.padding],
            dilation=[self.parameters.pool4.dilation, self.parameters.pool4.dilation],
        )
        x = self.conv4(x)
        print(x.memory_config)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.parameters.pool5.batch_size,
            input_h=self.parameters.pool5.input_height,
            input_w=self.parameters.pool5.input_width,
            channels=self.parameters.conv4.out_channels,
            kernel_size=[self.parameters.pool5.kernel_size, self.parameters.pool5.kernel_size],
            stride=[self.parameters.pool5.stride, self.parameters.pool5.stride],
            padding=[self.parameters.pool5.padding, self.parameters.pool5.padding],
            dilation=[self.parameters.pool5.dilation, self.parameters.pool5.dilation],
        )
        x = ttnn.reshape(x, [x.shape[0], -1])

        self.fc1_weights = preprocess_linear_weight(self.fc1_weights, self.device)
        self.fc2_weights = preprocess_linear_weight(self.fc2_weights, self.device)
        self.fc1_bias = preprocess_linear_bias(self.fc1_bias, self.device)
        self.fc2_bias = preprocess_linear_bias(self.fc2_bias, self.device)

        x = ttnn.linear(x, self.fc1_weights, bias=self.fc1_bias)

        x = ttnn.relu(x)

        x = ttnn.linear(x, self.fc2_weights, bias=self.fc2_bias)

        return x
