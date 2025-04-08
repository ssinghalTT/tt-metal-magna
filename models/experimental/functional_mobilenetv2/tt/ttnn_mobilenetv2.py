# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_mobilenetv2.tt.common import MobileNetV2Conv2D


class MobileNetV2:
    def __init__(self, model_params, device, batchsize) -> None:
        self.device = device
        self.model_parameters = model_params
        self.batchsize = batchsize

        self.c1 = MobileNetV2Conv2D(
            [3, 2, 1, 32], model_params[1], device, batchsize, use_shallow_covariant=True, deallocate_activation=True
        )
        self.c2 = MobileNetV2Conv2D([3, 1, 1, 32], model_params[2], device, batchsize, groups=32)
        self.c3 = MobileNetV2Conv2D([1, 1, 0, 16], model_params[3], device, batchsize)
        self.c4 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[4], device, batchsize)
        self.c5 = MobileNetV2Conv2D([3, 2, 1, 96], model_params[5], device, batchsize, groups=96)
        self.c6 = MobileNetV2Conv2D([1, 1, 0, 24], model_params[6], device, batchsize, deallocate_activation=False)
        self.c7 = MobileNetV2Conv2D([1, 1, 0, 144], model_params[7], device, batchsize)
        self.c8 = MobileNetV2Conv2D([3, 1, 1, 144], model_params[8], device, batchsize, groups=144)
        self.c9 = MobileNetV2Conv2D([1, 1, 0, 24], model_params[9], device, batchsize, deallocate_activation=False)
        self.c10 = MobileNetV2Conv2D([1, 1, 0, 144], model_params[10], device, batchsize)
        self.c11 = MobileNetV2Conv2D([3, 2, 1, 144], model_params[11], device, batchsize, groups=144)
        self.c12 = MobileNetV2Conv2D([1, 1, 0, 32], model_params[12], device, batchsize)
        self.c13 = MobileNetV2Conv2D([1, 1, 0, 192], model_params[13], device, batchsize)
        self.c14 = MobileNetV2Conv2D([3, 1, 1, 192], model_params[14], device, batchsize, groups=192)
        self.c15 = MobileNetV2Conv2D([1, 1, 0, 32], model_params[15], device, batchsize)
        self.c16 = MobileNetV2Conv2D([1, 1, 0, 192], model_params[16], device, batchsize)
        self.c17 = MobileNetV2Conv2D([3, 1, 1, 192], model_params[17], device, batchsize, groups=192)
        self.c18 = MobileNetV2Conv2D([1, 1, 0, 32], model_params[18], device, batchsize)
        self.c19 = MobileNetV2Conv2D([1, 1, 0, 192], model_params[19], device, batchsize)
        self.c20 = MobileNetV2Conv2D([3, 2, 1, 192], model_params[20], device, batchsize, groups=192, block_shard=True)
        self.c21 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[21], device, batchsize, block_shard=True)
        self.c22 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[22], device, batchsize, block_shard=True)
        self.c23 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[23], device, batchsize, groups=384, block_shard=True)
        self.c24 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[24], device, batchsize, block_shard=True)
        self.c25 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[25], device, batchsize, block_shard=True)
        self.c26 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[26], device, batchsize, groups=384, block_shard=True)
        self.c27 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[27], device, batchsize, block_shard=True)
        self.c28 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[28], device, batchsize, block_shard=True)
        self.c29 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[29], device, batchsize, groups=384, block_shard=True)
        self.c30 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[30], device, batchsize, block_shard=True)
        self.c31 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[31], device, batchsize, block_shard=True)
        self.c32 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[32], device, batchsize, groups=384, block_shard=True)
        self.c33 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[33], device, batchsize)
        self.c34 = MobileNetV2Conv2D([1, 1, 0, 576], model_params[34], device, batchsize, block_shard=True)
        self.c35 = MobileNetV2Conv2D([3, 1, 1, 576], model_params[35], device, batchsize, groups=576, block_shard=True)
        self.c36 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[36], device, batchsize)
        self.c37 = MobileNetV2Conv2D([1, 1, 0, 576], model_params[37], device, batchsize)
        self.c38 = MobileNetV2Conv2D([3, 1, 1, 576], model_params[38], device, batchsize, groups=576, block_shard=True)
        self.c39 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[39], device, batchsize)
        self.c40 = MobileNetV2Conv2D([1, 1, 0, 576], model_params[40], device, batchsize)
        self.c41 = MobileNetV2Conv2D([3, 2, 1, 576], model_params[41], device, batchsize, groups=576, block_shard=True)
        self.c42 = MobileNetV2Conv2D([1, 1, 0, 160], model_params[42], device, batchsize)
        self.c43 = MobileNetV2Conv2D([1, 1, 0, 960], model_params[43], device, batchsize)
        self.c44 = MobileNetV2Conv2D([3, 1, 1, 960], model_params[44], device, batchsize, groups=960, block_shard=True)
        self.c45 = MobileNetV2Conv2D([1, 1, 0, 160], model_params[45], device, batchsize)
        self.c46 = MobileNetV2Conv2D([1, 1, 0, 960], model_params[46], device, batchsize)
        self.c47 = MobileNetV2Conv2D([3, 1, 1, 960], model_params[47], device, batchsize, groups=960, block_shard=True)
        self.c48 = MobileNetV2Conv2D([1, 1, 0, 160], model_params[48], device, batchsize)
        self.c49 = MobileNetV2Conv2D([1, 1, 0, 960], model_params[49], device, batchsize)
        self.c50 = MobileNetV2Conv2D([3, 1, 1, 960], model_params[50], device, batchsize, groups=960, block_shard=True)
        self.c51 = MobileNetV2Conv2D([1, 1, 0, 320], model_params[51], device, batchsize, block_shard=True)
        self.c52 = MobileNetV2Conv2D([1, 1, 0, 1280], model_params[52], device, batchsize, block_shard=True)
        self.l1_weight = model_params["l1"]["weight"]
        self.l1_bias = model_params["l1"]["bias"]

    def __call__(self, x):
        output_tensor, h, w = self.c1(x)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c2(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c3(output_tensor)

        output_tensor, h, w = self.c4(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c5(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c6(output_tensor)

        output_tensor_c6 = output_tensor

        output_tensor, h, w = self.c7(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c8(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c9(output_tensor)

        output_tensor = ttnn.add(output_tensor_c6, output_tensor)
        output_tensor, h, w = self.c10(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c11(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c12(output_tensor)
        output_tensor_c12 = output_tensor

        output_tensor, h, w = self.c13(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c14(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c15(output_tensor)
        output_tensor_c15 = output_tensor
        output_tensor = output_tensor_c15 + output_tensor_c12
        output_tensor_a2 = output_tensor

        output_tensor, h, w = self.c16(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c17(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c18(output_tensor)
        output_tensor = output_tensor_a2 + output_tensor

        output_tensor, h, w = self.c19(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c20(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c21(output_tensor)
        output_tensor_c21 = output_tensor

        output_tensor, h, w = self.c22(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c23(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c24(output_tensor)
        output_tensor = output_tensor_c21 + output_tensor
        output_tensor_a4 = output_tensor

        output_tensor, h, w = self.c25(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c26(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c27(output_tensor)
        output_tensor = output_tensor_a4 + output_tensor
        output_tensor_a5 = output_tensor

        output_tensor, h, w = self.c28(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c29(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c30(output_tensor)
        output_tensor = ttnn.add(output_tensor_a5, output_tensor)

        output_tensor, h, w = self.c31(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c32(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c33(output_tensor)
        output_tensor_c33 = output_tensor

        output_tensor, h, w = self.c34(output_tensor_c33)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c35(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c36(output_tensor)
        output_tensor = output_tensor_c33 + output_tensor
        output_tensor_a7 = output_tensor

        output_tensor, h, w = self.c37(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c38(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c39(output_tensor)
        output_tensor = output_tensor_a7 + output_tensor

        output_tensor, h, w = self.c40(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c41(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c42(output_tensor)
        output_tensor_c42 = output_tensor

        output_tensor, h, w = self.c43(output_tensor_c42)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c44(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c45(output_tensor)
        output_tensor = output_tensor_c42 + output_tensor
        output_tensor_a9 = output_tensor

        output_tensor, h, w = self.c46(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c47(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c48(output_tensor)
        output_tensor = output_tensor + output_tensor_a9

        output_tensor, h, w = self.c49(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c50(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c51(output_tensor)

        output_tensor, h, w = self.c52(output_tensor)

        output_tensor = ttnn.relu6(output_tensor)
        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (self.batchsize, h, w, output_tensor.shape[3]))
        output_tensor = ttnn.global_avg_pool2d(output_tensor)

        output_tensor = ttnn.reshape(output_tensor, (self.batchsize, -1))

        output_tensor = ttnn.linear(output_tensor, self.l1_weight, bias=self.l1_bias)

        return output_tensor
