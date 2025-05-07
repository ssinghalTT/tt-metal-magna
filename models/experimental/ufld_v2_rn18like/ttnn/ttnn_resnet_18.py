# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.ufld_v2_rn18like.ttnn.common import TtnnUFLDv2RN18Conv2D
from models.experimental.ufld_v2_rn18like.ttnn.ttnn_basic_block import TtnnBasicBlock


def p(x, b="x"):
    print(f"{b}'s shape is {x.shape}")
    print(f"{b}'s layout is {x.layout}")
    print(f"{b}'s dtype is {x.dtype}")
    print(f"{b}'s config is {x.memory_config()}")


class TtnnResnet18:
    def __init__(self, conv_args, conv_pth, device):
        self.maxpool_args = conv_args.maxpool
        self.device = device
        self.conv1 = TtnnUFLDv2RN18Conv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=self.device,
            activation="relu",
            is_blk=False,
            spcl_case=True,
            is_dealloc_act=True,
        )
        # layer-1
        self.layer1_0 = TtnnBasicBlock(conv_args.layer1[0], conv_pth.layer1_0, device=self.device, is_downsample=False)
        self.layer1_1 = TtnnBasicBlock(conv_args.layer1[1], conv_pth.layer1_1, device=self.device, is_downsample=False)
        # layer-2
        self.layer2_0 = TtnnBasicBlock(conv_args.layer2[0], conv_pth.layer2_0, device=self.device, is_downsample=True)
        self.layer2_1 = TtnnBasicBlock(conv_args.layer2[1], conv_pth.layer2_1, device=self.device, is_downsample=False)
        # layer-3
        self.layer3_0 = TtnnBasicBlock(
            conv_args.layer3[0], conv_pth.layer3_0, device=self.device, is_downsample=True, blk_sharded=True
        )
        self.layer3_1 = TtnnBasicBlock(
            conv_args.layer3[1], conv_pth.layer3_1, device=self.device, is_downsample=False, blk_sharded=True
        )
        # layer-4
        self.layer4_0 = TtnnBasicBlock(
            conv_args.layer4[0], conv_pth.layer4_0, device=self.device, is_downsample=True, blk_sharded=True
        )
        self.layer4_1 = TtnnBasicBlock(
            conv_args.layer4[1], conv_pth.layer4_1, device=self.device, is_downsample=False, blk_sharded=True
        )

    def __call__(self, x, batch_size=1):
        x1, out_ht, out_wdth = self.conv1(x)
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
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)

        return x
