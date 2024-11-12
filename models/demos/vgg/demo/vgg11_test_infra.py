# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import torch
import torchvision

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
    _nearest_y,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
import transformers
from transformers import AutoImageProcessor
from torchvision import models
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.vgg.tt import ttnn_vgg
from models.demos.vgg.demo_utils import get_data, get_data_loader, get_batch, preprocess


class Vgg11TestInfra:
    def __init__(
        self,
        device,
        batch_size,
        input_loc,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        output_mesh_composer=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.output_mesh_composer = output_mesh_composer
        self.input_loc = input_loc

        self.model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        }

        # model_name = "vgg11"
        model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            convert_to_ttnn=lambda *_: True,
            custom_preprocessor=ttnn_vgg.custom_preprocessor,
        )
        self.model_state_dict = model.state_dict()

        ## IMAGENET INFERENCE
        # data_loader = get_data_loader(self.input_loc, batch_size, 1)
        # self.torch_input_tensor, labels = get_batch(data_loader)

        input_shape = (1, 224, 224, 3)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        self.input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        self.torch_output_tensor = model(self.torch_input_tensor)

    def run(self, tt_input_tensor=None):
        self.output_tensor = None
        self.output_tensor = ttnn_vgg.ttnn_vgg11(
            self.device,
            self.input_tensor,
            batch_size=self.batch_size,
            parameters=self.parameters,
            model_config=self.model_config,
        )
        return self.output_tensor

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        # sharded mem config for fold input

        # ===
        # shard_grid = ttnn.CoreRangeSet(
        #     {
        #         ttnn.CoreRange(
        #             ttnn.CoreCoord(0, 0),
        #             ttnn.CoreCoord(7, 0),
        #         ),
        #     }
        # )
        # n_cores = 8
        # shard_spec = ttnn.ShardSpec(shard_grid, [n * h * w // n_cores, c], ttnn.ShardOrientation.ROW_MAJOR, False)
        # input_mem_config = ttnn.MemoryConfig(
        #     ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        # )
        # ====

        num_cores = core_grid.x * core_grid.y
        print("num_cores", num_cores)
        shard_h = (n * w * h + num_cores - 1) // num_cores
        print("shard_h", shard_h)
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        print("grid_coord", grid_coord)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        print("shard_grid", shard_grid)
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR, False)
        print("shard_spec", shard_spec)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        print("input_mem_config", input_mem_config)
        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        # torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        # tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
        # print("tt_inputs_host",tt_inputs_host.memory_config)
        # print("input_mem_config",input_mem_config)
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device)
        dram_grid_size = device.dram_grid_size()
        print("dram_grid_size", dram_grid_size)
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
                16,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        print("dram_shard_spec", dram_shard_spec)
        # dram_shard_spec = ttnn.ShardSpec(
        #     ttnn.CoreRangeSet(
        #         {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        #     ),
        #     [
        #         divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x),
        #         tt_inputs_host.shape[-1],
        #     ],
        #     ttnn.ShardOrientation.ROW_MAJOR,
        #     False,
        # )

        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )
        print("sharded_mem_config_DRAM", sharded_mem_config_DRAM)

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config


def create_test_infra(
    device,
    batch_size,
    input_loc,
    inputs_mesh_mapper=None,
    weights_mesh_mapper=None,
    output_mesh_composer=None,
):
    return Vgg11TestInfra(
        device,
        batch_size,
        input_loc,
        inputs_mesh_mapper,
        weights_mesh_mapper,
        output_mesh_composer,
    )
