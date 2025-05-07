# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def p(x, b="x"):  # for debugging
    print(f"{b}'s shape is {x.shape}")
    print(f"{b}'s layout is {x.layout}")
    print(f"{b}'s dtype is {x.dtype}")
    print(f"{b}'s config is {x.memory_config()}")


class TtnnUFLDv2RN18Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_blk=False,
        is_wdth=False,
        is_dealloc_act=False,
        spcl_case=False,
    ):
        self.spcl_case = spcl_case
        if is_blk:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        if is_wdth:
            shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.cache = cache
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        self.is_dealloc_act = is_dealloc_act
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=None if self.spcl_case else shard_layout,
            deallocate_activation=self.is_dealloc_act,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=False if spcl_case else True,
            activation=activation,
            input_channels_alignment=16 if spcl_case else 8,
        )
        if conv_pth.bias is not None:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x):
        input_height = self.conv.input_height
        input_width = self.conv.input_width
        batch_size = self.conv.batch_size
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x, output_height, output_width
