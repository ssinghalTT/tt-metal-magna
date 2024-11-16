# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 1, 14],
    ],
)
def test_tanh_range(device, shape):
    torch_input_tensor_a = torch.tensor(
        [
            [
                [
                    [
                        -1,
                        -2,
                        -3,
                        -0.5,
                        -1.5,
                        -2.5,
                        -3.5,
                        -3.75,
                        -0.8359375,
                        -3.359375,
                        -1.8828125,
                        -3.255,
                        -4,
                        -5,
                    ]
                ]
            ]
        ],
        dtype=torch.bfloat16,
    )
    torch_output_tensor = torch.tanh(torch_input_tensor_a)
    torch.set_printoptions(linewidth=200, threshold=10000, precision=10, sci_mode=False, edgeitems=17)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.tanh(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output_tensor = ttnn.to_torch(output_tensor)

    for i in range(1):  # Batch size
        for j in range(1):  # Channels
            for k in range(shape[-2]):  # Height
                for l in range(shape[-1]):  # Width
                    print(
                        f"{i}-{j}-{k}-{l} input: {torch_input_tensor_a[i][j][k][l]} \t TT_out: {output_tensor[i][j][k][l]} \t torch: {torch_output_tensor[i][j][k][l]} \n"
                    )

    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    print("pcc_msg", pcc_msg)  #  AssertionError: 0.966528114289078
    assert pcc
