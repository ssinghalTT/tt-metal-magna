# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtnnUFLDV2Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_blk=False,
    ):
        if is_blk:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
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
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            activation=activation,
            input_channels_alignment=8,
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
            conv_op_cache=self.cache,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x, output_height, output_width


class TtnnBasicBlock:
    def __init__(self, conv_args, conv_pth, device, is_downsample=False, blk_sharded=False):
        self.is_downsample = is_downsample

        self.conv1 = TtnnUFLDV2Conv2D(
            conv_args.conv1, conv_pth.conv1, device=device, activation="relu", is_blk=blk_sharded
        )
        self.conv2 = TtnnUFLDV2Conv2D(conv_args.conv2, conv_pth.conv2, device=device, activation="", is_blk=blk_sharded)
        if is_downsample:
            self.downsample = TtnnUFLDV2Conv2D(
                conv_args.downsample[0], conv_pth.downsample, device=device, activation="", is_blk=blk_sharded
            )

    def __call__(self, input):
        x_identity = input
        x, out_ht, out_wdth = self.conv1(input)
        x, out_ht, out_wdth = self.conv2(x)
        if self.is_downsample:
            x_identity, out_ht, out_wdth = self.downsample(input)
        if x_identity.is_sharded():
            if x.memory_config().shard_spec.grid != x_identity.memory_config().shard_spec.grid:
                memory_config_req = x.memory_config()
                memory_config_req.shard_spec.shape = x_identity.memory_config().shard_spec.shape
                memory_config_req.shard_spec.grid = x_identity.memory_config().shard_spec.grid
                x = ttnn.reshard(x, memory_config_req)
        x = ttnn.add(x, x_identity, memory_config=x.memory_config())
        x = ttnn.relu(x)
        ttnn.deallocate(x_identity)
        return x


class TtnnResnet34:
    def __init__(self, conv_args, conv_pth, device):
        self.maxpool_args = conv_args.maxpool
        self.device = device
        self.conv1 = TtnnUFLDV2Conv2D(conv_args.conv1, conv_pth.conv1, device=self.device, activation="relu")
        self.layer1_0 = TtnnBasicBlock(conv_args.layer1[0], conv_pth.layer1_0, device=self.device, is_downsample=False)
        self.layer1_1 = TtnnBasicBlock(conv_args.layer1[1], conv_pth.layer1_1, device=self.device, is_downsample=False)
        self.layer1_2 = TtnnBasicBlock(conv_args.layer1[2], conv_pth.layer1_2, device=self.device, is_downsample=False)
        self.layer2_0 = TtnnBasicBlock(conv_args.layer2[0], conv_pth.layer2_0, device=self.device, is_downsample=True)
        self.layer2_1 = TtnnBasicBlock(conv_args.layer2[1], conv_pth.layer2_1, device=self.device, is_downsample=False)
        self.layer2_2 = TtnnBasicBlock(conv_args.layer2[2], conv_pth.layer2_2, device=self.device, is_downsample=False)
        self.layer2_3 = TtnnBasicBlock(conv_args.layer2[3], conv_pth.layer2_3, device=self.device, is_downsample=False)
        self.layer3_0 = TtnnBasicBlock(
            conv_args.layer3[0], conv_pth.layer3_0, device=self.device, is_downsample=True, blk_sharded=True
        )
        self.layer3_1 = TtnnBasicBlock(
            conv_args.layer3[1], conv_pth.layer3_1, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer3_2 = TtnnBasicBlock(
            conv_args.layer3[2], conv_pth.layer3_2, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer3_3 = TtnnBasicBlock(
            conv_args.layer3[3], conv_pth.layer3_3, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer3_4 = TtnnBasicBlock(
            conv_args.layer3[4], conv_pth.layer3_4, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer3_5 = TtnnBasicBlock(
            conv_args.layer3[5], conv_pth.layer3_5, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer4_0 = TtnnBasicBlock(
            conv_args.layer4[0], conv_pth.layer4_0, device=self.device, is_downsample=True, blk_sharded=True
        )
        self.layer4_1 = TtnnBasicBlock(
            conv_args.layer4[1], conv_pth.layer4_1, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer4_2 = TtnnBasicBlock(
            conv_args.layer4[2], conv_pth.layer4_2, device=self.device, is_downsample=False, blk_sharded=True
        )

    def __call__(self, x, batch_size=1):
        x1, out_ht, out_wdth = self.conv1(x)
        if x.is_sharded():
            ttnn.deallocate(x)
            x1 = ttnn.reallocate(x1)
        x1 = ttnn.max_pool2d(
            x1,
            batch_size=batch_size,
            input_h=out_ht,
            input_w=out_wdth,
            channels=x.shape[-1],
            kernel_size=[self.maxpool_args.kernel_size, self.maxpool_args.kernel_size],
            stride=[self.maxpool_args.stride, self.maxpool_args.stride],
            padding=[self.maxpool_args.padding, self.maxpool_args.padding],
            dilation=[self.maxpool_args.dilation, self.maxpool_args.dilation],
        )
        x = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x1)
        x = ttnn.reallocate(x)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        return x


class TtnnUFLDv2:
    def __init__(self, conv_args, conv_pth, device):
        self.input_height = 320
        self.input_width = 800
        self.num_grid_row = 100
        self.num_cls_row = 56
        self.num_grid_col = 100
        self.num_cls_col = 41
        self.num_lane_on_row = 4
        self.num_lane_on_col = 4
        self.use_aux = False
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        self.input_height = self.input_height
        self.input_width = self.input_width
        self.input_dim = self.input_height // 32 * self.input_width // 32 * 8
        self.conv_pth = conv_pth
        self.res_model = TtnnResnet34(conv_args, conv_pth.res_model, device=device)
        self.pool = TtnnUFLDV2Conv2D(conv_args.pool, conv_pth.pool, activation="", device=device)

    def __call__(self, input, batch_size=1):
        fea = self.res_model(input, batch_size=batch_size)
        fea, out_h, out_w = self.pool(fea)
        if fea.is_sharded():
            fea = ttnn.sharded_to_interleaved(fea, ttnn.L1_MEMORY_CONFIG)
        fea = ttnn.permute(fea, (0, 1, 3, 2))
        fea = ttnn.reshape(fea, (fea.shape[0], fea.shape[1], 1, fea.shape[2] * fea.shape[3]))
        grid_size = (8, 8)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        shard_spec = ttnn.ShardSpec(shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        width_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
        )
        fea = ttnn.to_memory_config(fea, width_sharded_mem_config)
        out = ttnn.linear(
            fea,
            self.conv_pth.cls.linear_1.weight,
            bias=self.conv_pth.cls.linear_1.bias,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        out = ttnn.relu(out)
        out = ttnn.linear(
            out,
            self.conv_pth.cls.linear_2.weight,
            bias=self.conv_pth.cls.linear_2.bias,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        return out
