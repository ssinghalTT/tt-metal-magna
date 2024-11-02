# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from models.common.lightweightmodule import LightweightModule


class SyncTensor(torch.nn.Module):
    def __init__(self, num_devices):
        super().__init__()
        self.num_devices = num_devices

    def forward(self, mesh_tensor, mask):
        idx = torch.argmax(mask, dim=0)
        tensor = torch.select(mesh_tensor, dim=0, index=idx).unsqueeze(0)  # [1, *]
        out = torch.concat([tensor] * self.num_devices, dim=0)  # [num_devices, *]
        return out


class TtSyncTensor(LightweightModule):
    def __init__(self, device, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_devices = device.get_num_devices()

    def get_ttnn_inputs(self, input_tensor, mask):
        tt_input = ttnn.from_torch(
            input_tensor.unsqueeze(0),  # [1, num_devices, *]
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            mesh_mapper=ShardTensorToMesh(self.device, dim=1),
        )
        tt_mask = ttnn.from_torch(
            mask.unsqueeze(0),  # [1, num_devices, 1, 1]
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            mesh_mapper=ShardTensorToMesh(self.device, dim=1),
        )

        return tt_input, tt_mask

    def forward(self, mesh_tensor, mask):
        # Use the mask to clean up the multi-device input tensor
        tensor = mask * mesh_tensor

        # All reduce to sync/replicate tensors across devices
        scattered = ttnn.reduce_scatter(
            tensor,
            scatter_dim=3,  # last dim
            math_op=ttnn.ReduceType.Sum,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        out = ttnn.all_gather(scattered, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # last dim

        return out


@pytest.mark.parametrize(
    "input_shape",
    [
        (32, 1024),
        (32, 2048),
        (1024, 2048),
    ],
)
def test_sync_tensor(
    t3k_mesh_device,
    input_shape,
    function_level_defaults,
):
    ##### Test setup #####
    dtype = ttnn.bfloat16
    num_devices = t3k_mesh_device.get_num_devices()
    logger.info(f"num_devices: {num_devices}")

    ##### Set up inputs #####
    masks = torch.zeros(num_devices, num_devices, 1, 1).float()
    for i in range(num_devices):
        masks[i, i] = 1.0
    input_tensor = torch.randn((num_devices, *input_shape))

    ##### Set up the ops #####
    sync_tensor = SyncTensor(num_devices)
    tt_sync_tensor = TtSyncTensor(t3k_mesh_device, dtype)

    ##### Run the ops for all possible masks #####
    for mask in masks:
        ##### Pytorch #####
        pt_out = sync_tensor(input_tensor, mask)

        ##### TTNN #####
        tt_input, tt_mask = tt_sync_tensor.get_ttnn_inputs(input_tensor, mask)
        tt_out_ = tt_sync_tensor(tt_input, tt_mask)
        tt_out = ttnn.to_torch(tt_out_, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=1))

        ##### Compare #####
        passing, output = comp_pcc(pt_out, tt_out)
        logger.info(output)
        assert passing
