# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@pytest.mark.parametrize(
    "n, c, h, w, dim0, dim1",
    [
        (16, 128, 8, 256, 2, 3),
    ],
)
def test_resnet50_fold(device, n, c, h, w, dim0, dim1):
    torch.manual_seed(0)
    input_shape = (n, c, h, w)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    ## WH -> HW
    torch_output = torch_input.transpose(dim0, dim1)

    core_grid = ttnn.CoreGrid(y=8, x=8)
    mem_config = ttnn.create_sharded_memory_config(
        input_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    tt_output = ttnn.transpose(tt_input, dim0, dim1)
    tt_output = ttnn.to_torch(tt_output.cpu())

    assert_with_pcc(torch_output, tt_output, 0.9999)
