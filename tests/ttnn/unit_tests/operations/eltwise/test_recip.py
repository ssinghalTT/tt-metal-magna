# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_range_dtype,
)


@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
    ],
)
def test_recip(device, shapes):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.ones(shapes, dtype=torch.bfloat16) * 1.3125
    torch_output_tensor = torch.reciprocal(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.reciprocal(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print("tt", output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    print("torch", torch_output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_unary_composite_recip_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range_dtype(input_shapes, -3, 3, device, False, False, ttnn.bfloat8_b)

    output_tensor = ttnn.reciprocal(input_tensor1)
    output_tensor_rm = output_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    golden_function = ttnn.get_golden_function(ttnn.reciprocal)
    golden_tensor = golden_function(in_data1)

    # for i in range(1):            # Batch size
    #     for j in range(1):        # Channels
    #         for k in range(32):   # Height
    #             for l in range(32):  # Width
    #                 print(f"input: {in_data1[i][j][k][l]} \t tt: {output_tensor_rm[i][j][k][l]} \t torch: {golden_tensor[i][j][k][l]} \n")

    comp_pass = compare_pcc([output_tensor], [golden_tensor], pcc=0.99)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_recip_ttnn(input_shapes, device):
    # in_data1,  = data_gen_with_range_dtype(input_shapes, -3, 3, device, False, False, ttnn.bfloat8_b)
    # in_data1 = torch.tensor([[ 0.1250,  0.8106, -1.2041, -2.2332,  2.1228,  1.0111,  0.2019, -1.9699,
    #       0.3302, -2.8733, -0.0945,  1.2270,  2.1118,  0.9262, -1.1676,  2.5739],
    #     [ 0.8027, -0.3120,  0.5595, -1.4412, -1.9830,  2.9601, -1.7990,  1.6956,
    #       1.1888, -0.0658, -2.0195,  0.6662, -2.3045, -2.0631,  2.1113,  2.6135],
    #     [ 1.6186,  2.2592,  1.4965,  2.7283,  1.6544,  0.0987,  1.3307, -2.8272,
    #      -0.6272, -1.3947, -1.6155, -1.1754, -1.8809,  2.3578, -2.3728, -1.2367],
    #     [-2.8369,  1.1364,  1.6720, -2.4717,  0.4636, -2.6561,  2.0845,  1.6467,
    #       0.5338, -1.1267,  2.5902,  2.7318,  2.4500, -0.3181,  1.9408, -0.3645]], dtype=torch.bfloat16)
    in_data1 = torch.tensor(
        [
            [
                0.0000,
                0.0250,
                0.0500,
                0.0750,
                0.1000,
                0.1250,
                0.1500,
                0.1750,
                0.2000,
                0.2250,
                0.2500,
                0.2750,
                0.3000,
                0.3250,
                0.3500,
                0.3750,
            ],
            [
                0.4000,
                0.4250,
                0.4500,
                0.4750,
                0.5000,
                0.5250,
                0.5500,
                0.5750,
                0.6000,
                0.6250,
                0.6500,
                0.6750,
                0.0000,
                0.7250,
                0.7500,
                0.7750,
            ],
            [
                0.8000,
                0.8250,
                0.8500,
                0.8750,
                0.9000,
                0.9250,
                0.9500,
                0.9750,
                1.0000,
                1.0250,
                1.0500,
                1.0750,
                1.1000,
                1.1250,
                1.1500,
                1.1750,
            ],
            [
                1.2000,
                1.2250,
                1.2500,
                1.2750,
                1.3000,
                1.3250,
                1.3500,
                1.3750,
                1.4000,
                1.4250,
                1.4500,
                1.4750,
                1.5000,
                1.5250,
                1.5500,
                1.5750,
            ],
        ],
        dtype=torch.bfloat16,
    )
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    in_torch = ttnn.to_torch(input_tensor1)
    output_tensor = ttnn.reciprocal(input_tensor1)
    output_tensor_rm = ttnn.to_torch(output_tensor)

    golden_function = ttnn.get_golden_function(ttnn.reciprocal)
    golden_tensor = golden_function(in_torch)
    print(output_tensor_rm)
    print(golden_tensor)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor_rm)
    assert comp_pass >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_recip_ttnn2(input_shapes, device):
    in_data1 = torch.arange(-63.0, 1, 1.0, dtype=torch.bfloat16).reshape(4, 16)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    in_torch = ttnn.to_torch(input_tensor1)
    output_tensor = ttnn.reciprocal(input_tensor1)
    output_tensor_rm = ttnn.to_torch(output_tensor)

    golden_function = ttnn.get_golden_function(ttnn.reciprocal)
    golden_tensor = golden_function(in_torch)
    print(output_tensor_rm)
    print(golden_tensor)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor_rm)
    assert comp_pass >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_recip_input(input_shapes, device):
    # in_data1,  = data_gen_with_range_dtype(input_shapes, -3, 3, device, False, False, ttnn.bfloat8_b)
    in_data1 = torch.ones(input_shapes, dtype=torch.bfloat16) * 0.0987
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    in_torch = ttnn.to_torch(input_tensor1)
    output_tensor = ttnn.reciprocal(input_tensor1)
    output_tensor_rm = ttnn.to_torch(output_tensor)

    golden_function = ttnn.get_golden_function(ttnn.reciprocal)
    golden_tensor = golden_function(in_torch)

    # print(output_tensor_rm)
    # print(golden_tensor)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor_rm)
    assert comp_pass >= 0.99
