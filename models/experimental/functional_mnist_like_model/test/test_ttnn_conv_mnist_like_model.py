# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_mnist_like_model.reference import conv_mnist_like_model
from models.experimental.functional_mnist_like_model.ttnn import ttnn_conv_mnist_like_model
from models.experimental.functional_mnist_like_model.ttnn.model_preprocessing import (
    create_conv_mnist_like_model_input_tensors,
    create_conv_mnist_like_model_model_parameters,
)
import os


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_conv_mnist_like_model(device, reset_seeds):
    torch_input, ttnn_input = create_conv_mnist_like_model_input_tensors()
    torch_model = conv_mnist_like_model.Conv_mnist_like_model(1000)
    torch_model.eval()
    torch_output = torch_model(torch_input)

    parameters = create_conv_mnist_like_model_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_conv_mnist_like_model.Conv_mnist_like_model(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)
    # ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99999)
