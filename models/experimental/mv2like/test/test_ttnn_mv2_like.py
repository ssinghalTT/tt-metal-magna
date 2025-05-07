# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.experimental.mv2like.reference.mv2_like import Mv2Like
from models.experimental.mv2like.tt.model_preprocessing import (
    create_mv2_like_input_tensors,
    create_mv2_like_model_parameters,
)
from models.experimental.mv2like.tt.ttnn_mv2_like import TtMv2Like
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 8],
)
def test_mv2_like(device, batch_size, reset_seeds):
    weights_path = (
        "models/experimental/mv2like/lraspp_mobilenet_v2_trained_statedict.pth"  # specify your weights path here
    )

    state_dict = torch.load(weights_path)
    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = Mv2Like()
    new_state_dict = {
        name1: parameter2
        for (name1, parameter2), (name2, parameter1) in zip(torch_model.state_dict().items(), ds_state_dict.items())
        if isinstance(parameter2, torch.FloatTensor)
    }
    torch_model.load_state_dict(new_state_dict)

    torch_model = torch_model.eval()
    # SS
    torch_input_tensor, ttnn_input_tensor = create_mv2_like_input_tensors(
        batch=batch_size, input_height=224, input_width=224
    )

    torch_output_tensor = torch_model(torch_input_tensor)

    model_parameters = create_mv2_like_model_parameters(torch_model, device=device)

    model = TtMv2Like(model_parameters, device, batchsize=batch_size)

    output_tensor = model(ttnn_input_tensor)

    output_tensor = ttnn.to_torch(output_tensor).permute(0, 3, 1, 2)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
