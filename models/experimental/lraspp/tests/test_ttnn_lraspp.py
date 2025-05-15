# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.experimental.lraspp.reference.lraspp import LRASPP
from models.experimental.lraspp.tt.model_preprocessing import (
    create_lraspp_input_tensors,
    create_lraspp_model_parameters,
)
from models.experimental.lraspp.tt.ttnn_lraspp import TtLRASPP
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 8],
)
def test_lraspp(device, batch_size, reset_seeds):
    weights_path = (
        "models/experimental/lraspp/lraspp_mobilenet_v2_trained_statedict.pth"  # specify your weights path here
    )

    state_dict = torch.load(weights_path)
    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = LRASPP()
    new_state_dict = {
        name1: parameter2
        for (name1, parameter2), (name2, parameter1) in zip(torch_model.state_dict().items(), ds_state_dict.items())
        if isinstance(parameter2, torch.FloatTensor)
    }
    torch_model.load_state_dict(new_state_dict)

    torch_model = torch_model.eval()
    # SS
    torch_input_tensor, ttnn_input_tensor = create_lraspp_input_tensors(
        batch=batch_size, input_height=224, input_width=224
    )

    torch_output_tensor = torch_model(torch_input_tensor)

    model_parameters = create_lraspp_model_parameters(torch_model, device=device)

    model = TtLRASPP(model_parameters, device, batchsize=batch_size)

    output_tensor = model(ttnn_input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)[:, :, :, 0:1].permute(0, 3, 1, 2)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
