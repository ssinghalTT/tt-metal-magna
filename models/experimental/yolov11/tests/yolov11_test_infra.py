# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from models.experimental.yolov11.reference import yolov11
from models.experimental.yolov11.tt import ttnn_yolov11
from models.experimental.yolov11.reference.yolov11 import attempt_load
import sys
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)
from models.experimental.yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
import torch.nn as nn

try:
    sys.modules["ultralytics"] = yolov11
    sys.modules["ultralytics.nn.tasks"] = yolov11
    sys.modules["ultralytics.nn.modules.conv"] = yolov11
    sys.modules["ultralytics.nn.modules.block"] = yolov11
    sys.modules["ultralytics.nn.modules.head"] = yolov11

except KeyError:
    print("models.experimental.yolov11.reference.yolov11 not found.")


def load_yolov11_model():
    torch_model = attempt_load("yolov11n.pt", map_location="cpu")
    state_dict = torch_model.state_dict()
    torch_model = yolov11.YoloV11()
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model


class Yolov11TestInfra:
    def __init__(
        self,
        device,
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.model_location_generator = model_location_generator
        self.torch_input, self.ttnn_input = create_yolov11_input_tensors(device)
        torch_model = load_yolov11_model()
        parameters = create_yolov11_model_parameters(torch_model, self.torch_input, device=device)
        self.torch_output = torch_model(self.torch_input)
        self.ttnn_yolov11_model = ttnn_yolov11.YoloV11(device, parameters)

    def run(self):
        self.output_tensor = self.ttnn_yolov11_model(self.input_tensor)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input if self.torch_input is None else self.torch_input

        n, c, h, w = torch_input_tensor.shape
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_h = (n * w * h + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
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
                16,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(self.output_tensor)
        output_tensor = output_tensor.reshape((self.torch_output).shape)

        valid_pcc = 0.98
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output, output_tensor, pcc=valid_pcc)

        logger.info(f"Yolov11, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)


def create_test_infra(
    device,
    model_location_generator=None,
):
    return Yolov11TestInfra(
        device,
        model_location_generator,
    )
