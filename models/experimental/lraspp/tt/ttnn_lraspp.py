# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.lraspp.tt.common import TtConv2D, TtInvertedResidual, shard_upsample


class TtLRASPP:
    def __init__(self, model_params, device, batchsize) -> None:
        self.device = device
        self.model_parameters = model_params
        self.batchsize = batchsize

        self.conv1 = TtConv2D(
            [3, 2, 1, 32],
            (model_params["fused_conv_0_weight"], model_params["fused_conv_0_bias"]),
            device,
            batchsize,
            use_shallow_covariant=True,
            deallocate_activation=True,
        )
        self.conv2 = TtConv2D(
            [3, 1, 1, 32],
            (model_params["fused_conv_1_weight"], model_params["fused_conv_1_bias"]),
            device,
            batchsize,
            groups=32,
            deallocate_activation=True,
        )
        self.conv3 = TtConv2D(
            [1, 1, 0, 16], (model_params["conv_0_weight"], model_params["conv_0_bias"]), device, batchsize
        )

        self.block1 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=2,
            in_channels=16,
            out_channels=24,
            id=1,
            block_shard=False,
        )
        self.block2 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=24,
            out_channels=24,
            id=2,
            block_shard=False,
        )
        self.block3 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=2,
            in_channels=24,
            out_channels=32,
            id=3,
            block_shard=False,
        )
        self.block4 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=32,
            out_channels=32,
            id=4,
            block_shard=False,
        )
        self.block5 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=32,
            out_channels=32,
            id=5,
            block_shard=False,
        )
        self.block6 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=2,
            in_channels=32,
            out_channels=64,
            id=6,
            block_shard=True,
        )
        self.block7 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=64,
            out_channels=64,
            id=7,
            block_shard=True,
        )
        self.block8 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=64,
            out_channels=64,
            id=8,
            block_shard=True,
        )
        self.block9 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=64,
            out_channels=64,
            id=9,
            block_shard=True,
        )
        self.block10 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=64,
            out_channels=96,
            id=10,
            block_shard=True,
        )
        self.block11 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=96,
            out_channels=96,
            id=11,
            block_shard=True,
        )
        self.block12 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=96,
            out_channels=96,
            id=12,
            block_shard=True,
        )
        self.block13 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=2,
            in_channels=96,
            out_channels=160,
            id=13,
            block_shard=True,
        )
        self.block14 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=160,
            out_channels=160,
            id=14,
            block_shard=True,
        )
        self.block15 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=160,
            out_channels=160,
            id=15,
            block_shard=True,
        )
        self.block16 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=160,
            out_channels=320,
            id=16,
            block_shard=True,
        )

        self.conv4 = TtConv2D(
            [1, 1, 0, 1280],
            (model_params["fused_conv_34_weight"], model_params["fused_conv_34_bias"]),
            device,
            batchsize,
            block_shard=True,
        )

        self.conv5 = TtConv2D([1, 1, 0, 128], model_params[53], device, batchsize, width_shard=True)
        self.conv6 = TtConv2D([1, 1, 0, 128], model_params[54], device, batchsize, width_shard=True)
        self.low_classifier = TtConv2D(
            [1, 1, 0, 1],
            model_params["low_classifier"],
            device,
            batchsize,
            width_shard=True if batchsize <= 6 else False,
        )
        self.high_classifier = TtConv2D(
            [1, 1, 0, 1],
            model_params["high_classifier"],
            device,
            batchsize,
            width_shard=True if batchsize <= 6 else False,
        )

    def __call__(self, input):
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

        output_tensor, h, w = self.conv1(input)
        output_tensor = ttnn.relu6(output_tensor)
        ttnn.deallocate(input)
        ttnn.deallocate(nhwc)
        output_tensor = ttnn.reallocate(output_tensor)
        output_tensor, h, w = self.conv2(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor, h, w = self.conv3(output_tensor)
        output_tensor = self.block1(output_tensor)
        output_tensor = self.block2(output_tensor)
        output_tensor_3 = self.block3(output_tensor)
        output_tensor = self.block4(output_tensor_3)
        output_tensor = self.block5(output_tensor)
        output_tensor = self.block6(output_tensor)
        output_tensor = self.block7(output_tensor)
        output_tensor = self.block8(output_tensor)
        output_tensor = self.block9(output_tensor)
        output_tensor = self.block10(output_tensor)
        output_tensor = self.block11(output_tensor)
        output_tensor = self.block12(output_tensor)
        output_tensor = self.block13(output_tensor)
        output_tensor = self.block14(output_tensor)
        output_tensor = self.block15(output_tensor)
        output_tensor = self.block16(output_tensor)

        output_tensor_1, h1, w1 = self.conv4(output_tensor)
        output_tensor_1 = ttnn.relu6(output_tensor_1)

        output_tensor_conv, h2, w2 = self.conv5(output_tensor_1)
        output_tensor_conv = ttnn.relu(output_tensor_conv)

        if output_tensor_1.is_sharded():
            output_tensor_1 = ttnn.sharded_to_interleaved(output_tensor_1, ttnn.L1_MEMORY_CONFIG)
        output_tensor_1 = ttnn.to_layout(output_tensor_1, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor_1 = ttnn.reshape(output_tensor_1, (self.batchsize, h1, w1, output_tensor_1.shape[3]))

        output_tensor = ttnn.global_avg_pool2d(output_tensor_1)

        output_tensor_2, h, w = self.conv6(output_tensor)
        output_tensor_2 = ttnn.sigmoid(output_tensor_2)

        output_tensor_2 = ttnn.sharded_to_interleaved(output_tensor_2, ttnn.L1_MEMORY_CONFIG)
        output_tensor_conv = ttnn.sharded_to_interleaved(output_tensor_conv, ttnn.L1_MEMORY_CONFIG)
        output_tensor_2 = ttnn.reshape(output_tensor_2, (self.batchsize, h, w, output_tensor_2.shape[3]))
        output_tensor_conv = ttnn.reshape(output_tensor_conv, (self.batchsize, h2, w2, output_tensor_conv.shape[3]))
        output_tensor = output_tensor_2 * output_tensor_conv

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        low_classifier, h, w = self.low_classifier(output_tensor_3)
        resize = shard_upsample(output_tensor, scale=4)
        high_classifier, h, w = self.high_classifier(resize)
        output_tensor = ttnn.add(high_classifier, low_classifier, dtype=ttnn.bfloat16)
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (self.batchsize, h, w, output_tensor.shape[3]))
        output_tensor = ttnn.pad(output_tensor, [(0, 7)], value=0.0)
        resize = shard_upsample(output_tensor, scale=8)
        return resize
        # output_tensor = ttnn.sharded_to_interleaved(resize, memory_config=ttnn.L1_MEMORY_CONFIG)
        # output_tensor = output_tensor[:, :, :, 0:1]

        # return output_tensor
