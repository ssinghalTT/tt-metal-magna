# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import untilize, comp_pcc
from models.utility_functions import is_grayskull, skip_for_blackhole
import ipdb
from collections import Counter


def find_first_non_equal_row(golden_res, our_res):
    eqs = torch.isclose(golden_res, our_res)
    row_statuses = eqs.all(dim=1)
    for row, state in enumerate(row_statuses):
        if not state:
            return row
    return None


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        [
            (1, 1, 87 * 2, 1),  # 2 tiles per core -> working on BH
            (1, 1, 348, 1),  # 4 tile height per core -> not working
            (1, 1, 261, 1),  # 3 tiles per core -> smallest failing case for bh untilize
            (1, 1, 1, 2),
            (5, 2, 4, 8),
            (5, 2, 4, 7),
            ## resnet shapes
            (1, 1, 1, 1),
            (1, 1, 7, 8),
            (1, 1, 49, 1),
            (1, 1, 49, 16),
            (1, 1, 49, 32),
            (1, 1, 196, 4),
            (1, 1, 196, 8),
            (1, 1, 196, 16),
            (1, 1, 784, 2),
            (1, 1, 784, 4),
            (1, 1, 784, 8),
            (1, 1, 3136, 2),
        ]
    ),
)
def test_run_untilize_test(dtype, nb, nc, nh, nw, device):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    shape = [nb, nc, 32 * nh, 32 * nw]

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=3000, threshold=10000, edgeitems=128)

    torch.manual_seed(10)

    if dtype == ttnn.float32:
        inp = torch.rand(*shape).float() * 1000.0
    else:
        inp = torch.rand(*shape).bfloat16()

    a = ttnn.Tensor(
        inp.flatten().tolist(),
        shape,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
    )

    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    b1 = ttnn.untilize(a, memory_config=out_mem_config, use_multicore=True, use_pack_untilize=True)
    c1 = ttnn.to_torch(b1)

    untilized_inp = untilize(inp)

    if dtype == ttnn.float32:
        passing1, output = comp_pcc(untilized_inp, c1, 0.999999)
        logger.info(output)
    else:
        passing1 = torch.equal(untilized_inp, c1)

    our_res = c1.squeeze()
    golden_res = untilized_inp.squeeze()

    first_non_equal_row = find_first_non_equal_row(golden_res, our_res)
    if first_non_equal_row is not None:
        logger.error(f"First non-equal row: {first_non_equal_row}")
        logger.error(f"Golden result: {golden_res[first_non_equal_row]}")
        logger.error(f"Our result: {our_res[first_non_equal_row]}")

    ipdb.set_trace()

    assert passing1
