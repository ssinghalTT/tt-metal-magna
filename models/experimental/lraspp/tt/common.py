# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math


# def determine_num_cores_for_upsample(nhw: int, width: int, max_cores=64) -> int:
def determine_num_cores_for_upsample(nhw: int, width: int, max_cores=16) -> int:
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores > max_cores:
        for divisor in range(max_cores, 0, -1):
            if nhw % divisor == 0 and (nhw // divisor) % width == 0:
                cores = divisor
                break
    return cores


def get_core_grid_from_num_cores(num_cores: int, grid_rows: int = 8, grid_cols: int = 8):
    rows = num_cores // grid_cols
    assert rows <= grid_rows, "Not enough cores for specified core grid"
    ranges = []
    if rows != 0:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_rows - 1, rows - 1),
            )
        )
    remainder = num_cores % grid_rows
    if remainder != 0:
        assert rows + 1 <= grid_rows, "Not enough cores for specified core grid"
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, rows),
                ttnn.CoreCoord(remainder - 1, rows),
            )
        )
    return ttnn.CoreRangeSet({*ranges})


def shard_upsample(x, scale=4):
    nhw = x.shape[0] * x.shape[1] * x.shape[2]
    num_cores = determine_num_cores_for_upsample(nhw, x.shape[2])
    core_grid = get_core_grid_from_num_cores(num_cores)
    shardspec = ttnn.create_sharded_memory_config_(
        x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    if x.is_sharded():
        x = ttnn.reshard(x, shardspec)
    else:
        x = ttnn.interleaved_to_sharded(x, shardspec)
    x = ttnn.upsample(x, (scale, scale), memory_config=x.memory_config())
    return x


class TtConv2D:
    def __init__(
        self,
        input_params,
        parameters,
        device,
        batch_size,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        width_shard=False,
        act_blocks=32,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        reshard_if_not_optimal=False,
        use_shallow_covariant=False,
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.device = device
        self.parameters = parameters
        self.activation_dtype = activation_dtype
        self.weights_dtype = weights_dtype
        self.input_params = input_params
        self.groups = groups
        self.dilation = dilation
        self.act_block_h = act_block_h
        self.block_shard = block_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout
        self.width_shard = width_shard
        self.act_blocks = act_blocks
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_split_reader = enable_split_reader
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.batch_size = batch_size
        self.shard_layout = shard_layout
        if self.block_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        if self.width_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        self.use_shallow_covariant = use_shallow_covariant
        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            dtype=self.activation_dtype,
            weights_dtype=self.weights_dtype,
            activation="",
            shard_layout=self.shard_layout,
            input_channels_alignment=16 if self.use_shallow_covariant else 32,
            act_block_w_div=1,
            transpose_shards=False,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_split_reader=self.enable_split_reader,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=True,
        )

        if self.act_block_h:
            conv_config.act_block_h_override = self.act_blocks

        if self.block_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        if x.shape[1] != 1 or x.shape[2] == 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            input_height = int(math.sqrt((x.shape[2] // self.batch_size)))
            input_width = int(math.sqrt((x.shape[2] // self.batch_size)))
        [x, [h, w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=x.shape[3],
            out_channels=self.input_params[3],
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=(self.input_params[0], self.input_params[0]),
            stride=(self.input_params[1], self.input_params[1]),
            padding=(self.input_params[2], self.input_params[2]),
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        return x, h, w


class TtInvertedResidual:
    def __init__(
        self, model_params, device, batchsize, expand_ratio, stride, in_channels, out_channels, id, block_shard=False
    ):
        self.device = device
        self.batchsize = batchsize
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_shard = block_shard
        self.id = id
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.conv1 = None
        if expand_ratio != 1:
            self.conv1 = TtConv2D(
                [1, 1, 0, hidden_dim],
                (model_params[f"fused_conv_{id * 3 - id}_weight"], model_params[f"fused_conv_{id * 3 - id}_bias"]),
                device,
                batchsize,
                block_shard=False if id == 6 and (11 < id <= 16) else self.block_shard,
            )

        self.conv2 = TtConv2D(
            [3, stride, 1, hidden_dim],
            (model_params[f"fused_conv_{id * 3 -id +1}_weight"], model_params[f"fused_conv_{id * 3 - id + 1}_bias"]),
            device,
            batchsize,
            groups=hidden_dim,
            block_shard=self.block_shard,
        )
        self.conv3 = TtConv2D(
            [1, 1, 0, out_channels],
            (model_params[f"conv_{id}_weight"], model_params[f"conv_{id}_bias"]),
            device,
            batchsize,
            block_shard=False if (10 <= id <= 16) else self.block_shard,
        )

    def __call__(self, x):
        identity = x
        if self.conv1 is not None:
            x, h, w = self.conv1(x)
            x = ttnn.relu6(x)
            # x = out
        out, h, w = self.conv2(x)
        ttnn.deallocate(x)
        out = ttnn.relu6(out)
        out, h, w = self.conv3(out)
        if self.use_res_connect:
            return ttnn.add(identity, out)
        return out
