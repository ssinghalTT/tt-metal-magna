# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.utility_functions import is_close


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x",
    [
        # (1, 768, 1, 1024, 32, 1, 8, 8),  # test batch size 9 (uneven batch sizes)
        # (1, 768, 1, 512, 32, 2, 8, 8),  # test group channel count is less than tile size
        # (1, 640, 128, 128, 32, 2, 4, 8),  # Stable Diffusion XL Variant 1
        # (1, 960, 128, 128, 32, 2, 2, 8),  # Stable Diffusion XL Variant 2
        # (11, 512, 120, 212, 32, 10, 8, 8),
        (2, 480, 1, 32, 8, 1, 1, 2)
        # (8, 480, 1, 32, 8, 1, 1, 8 )
        # (1, 1920, 1, 4096, 32, 16, 4, 8), # test batch size 8 (no multicast)
        # (1, 480, 64, 64, 8, 10, 1, 1), # test batch size 8 (no multicast)
        # (1 , 480, 64, 64, 8, 16, 1, 2), # test batch size 8 (no multicast)
    ],
)
def test_group_norm_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.ones((C,), dtype=torch.bfloat16)
    torch_bias = torch.zeros((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_row_major = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # groupnorm
    output_tensor = ttnn.group_norm(
        input_tensor_tilized,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=num_out_blocks,
    )

    # output tensor
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # print(output_tensor.shape)
    print(input_tensor.shape)

    # Write full tensor to a file
    torch.set_printoptions(profile="full")
    # copy = copy.t()
    # print(copy.shape)
    with open("tensor_input.txt", "w") as f:
        f.write(str(input_tensor[1][0][0][0:60]))
        f.write("\n")
    with open("tensor_output_partial.txt", "w") as f:
        f.write(str(output_tensor[1][0][0][0:60]))
    with open("tensor_output.txt", "w") as f:
        f.write(str(output_tensor))
    diff = torch.abs(torch_output_tensor - output_tensor)
    with open("diff_tensor.txt", "w") as f:
        f.write(str(diff))

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9996)
