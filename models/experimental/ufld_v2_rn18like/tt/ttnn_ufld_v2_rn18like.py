# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.ufld_v2_rn18like.tt.common import TtnnUFLDv2RN18Conv2D
from models.experimental.ufld_v2_rn18like.tt.ttnn_resnet_18 import TtnnResnet18


class TtnnUFLDV2RN18like:
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
        self.res_model = TtnnResnet18(conv_args, conv_pth.model, device=device)
        self.pool = TtnnUFLDv2RN18Conv2D(conv_args.pool, conv_pth.pool, activation="", device=device)
        self.last_conv = TtnnUFLDv2RN18Conv2D(conv_args[1], conv_pth.cls.conv_1, activation="", device=device)

    def __call__(self, input, batch_size=1):
        # Input pre-processing
        N, C, H, W = input.shape
        min_channels = 16
        if C < min_channels:
            channel_padding_needed = min_channels - C
            nchw = ttnn.pad(input, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = input
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))  # NCHW -> NHWC
        ttnn.deallocate(nchw)
        ttnn.deallocate(input)
        nhwc = ttnn.reallocate(nhwc)
        input = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])
        # ttnn.deallocate(nhwc)

        # ResNet mode
        fea = self.res_model(input, batch_size=batch_size)
        fea, out_h, out_w = self.pool(fea)
        fea = ttnn.to_layout(fea, ttnn.ROW_MAJOR_LAYOUT)
        fea, _, _ = self.last_conv(fea)
        fea = ttnn.relu(fea)
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
        return out
