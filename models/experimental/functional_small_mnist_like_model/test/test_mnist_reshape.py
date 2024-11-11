# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("permute", [True, False])
def test_reshape_small_mnist(device, permute):
    torch_input_tensor = torch.randn(2, 64, 3, 3)
    torch_result = torch_input_tensor.reshape(torch_input_tensor.shape[0], -1)

    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        ttnn_input_tensor.shape[0],
        1,
        ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, device=device)
    input_tensor = ttnn.reshape(input_tensor, [2, 3, 3, 64])
    if permute:
        input_tensor = ttnn.permute(input_tensor, (0, 3, 1, 2))
    ttnn_output = ttnn.reshape(input_tensor, (2, -1))

    output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_result, output, 0.9999)
