# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.lraspp.reference.lraspp import LRASPP
from models.experimental.lraspp.tt.model_preprocessing import (
    create_lraspp_model_parameters,
)
from loguru import logger
from models.experimental.lraspp.tt.ttnn_lraspp import TtLRASPP
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    is_wormhole_b0,
    divup,
)


def load_torch_model():
    weights_path = (
        "models/experimental/lraspp/lraspp_mobilenet_v2_trained_statedict.pth"  # specify your weights path here
    )

    state_dict = torch.load(weights_path)
    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = LRASPP()
    new_state_dict = {
        name1: parameter2
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
        if isinstance(parameter2, torch.FloatTensor)
    }
    torch_model.load_state_dict(new_state_dict)

    torch_model.eval()

    return torch_model


def load_ttnn_model(device, torch_model, batch_size):
    model_parameters = create_lraspp_model_parameters(torch_model, device=device)
    model = TtLRASPP(model_parameters, device, batchsize=batch_size)

    return model


class lrasppTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.model_location_generator = model_location_generator
        torch_model = load_torch_model()
        self.ttnn_lraspp_model = load_ttnn_model(
            device=self.device, torch_model=torch_model, batch_size=self.batch_size
        )
        input_shape = (batch_size, 224, 224, 3)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        self.input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

    def run(self):
        self.output_tensor = self.ttnn_lraspp_model(self.input_tensor)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        if c == 3:
            c = 16
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=7),
            ttnn.ShardStrategy.HEIGHT,
        )

        # sharded mem config for fold input
        # num_cores = core_grid.x * core_grid.y
        # shard_h = (n * w * h + num_cores - 1) // num_cores
        # grid_size = core_grid
        # grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        # shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        # shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
        # input_mem_config = ttnn.MemoryConfig(
        #    ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        # )
        # torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        # torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        # tt_inputs_host = torch.nn.functional.pad(torch_input_tensor, (0, 13), "constant", 0)

        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        # tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        # tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)

        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(self.output_tensor)[:, :, :, 0:1].permute(0, 3, 1, 2)

        valid_pcc = 0.98
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(f"lraspp batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)


def create_test_infra(
    device,
    batch_size,
):
    return lrasppTestInfra(
        device,
        batch_size,
    )
