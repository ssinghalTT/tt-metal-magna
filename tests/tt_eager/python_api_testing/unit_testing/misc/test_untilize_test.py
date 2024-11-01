# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import untilize, comp_pcc
from models.utility_functions import is_grayskull, skip_for_blackhole

# import ipdb
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
            # Look at the test output and verify that the core grid
            # size is 13x10. Notice the comments for the number of
            # blocks processed by the core for different shapes.
            # The block dimension is expressed in tiles, so 2x3 block
            # contains 6 32x32 tiles.
            (1, 1, 1, 1),  # 1 1x1 block, nblocks_per_core = 1 -> working on BH
            (1, 1, 32, 1),  # 32 1x1 blocks, nblocks_per_core = 1 -> working on BH
            (1, 1, 32, 32),  # 32 1x32 blocks, nblocks_per_core = 1 -> working on BH
            (1, 1, 130, 128),  # 130 1x128 blocks, nblocks_per_core = 1 -> working
            (1, 1, 131, 128),  # 131 1x128 blocks, nblocks_per_core = 2 -> working
            # Now, you can increase nw, but beware that at a certain point
            # you'll exceed core's L1 capacity or make Circular Buffers overlap.
            # All in all, nw dicatates block size so L1 is the constraint.
            (1, 1, 260, 1),  # 260 1x1 blocks, nblocks_per_core = 2 -> working
            (1, 1, 260, 3),  # 260 1x3 blocks, nblocks_per_core = 2 -> working
            (1, 1, 260, 5),  # 260 1x5 blocks, nblocks_per_core = 2 -> working
            (1, 1, 261, 1),  # 261 1x1 blocks, nblocks_per_core = 3 -> NOT working
            # Interesting, increasing block size causes the number of blocks
            # allocated to a core not to matter anymore, or at least sometimes
            (1, 1, 261, 3),  # 261 1x3 blocks, nblocks_per_core = 3 -> working
            (1, 1, 261, 5),  # 261 1x5 blocks, nblocks_per_core = 3 -> working
            (1, 1, 391, 5),  # 391 1x5 blocks, nblocks_per_core = 4 -> working
            (1, 1, 521, 5),  # 521 1x5 blocks, nblocks_per_core = 5 -> working
            (1, 1, 521, 3),  # 521 1x3 blocks, nblocks_per_core = 5 -> NOT working
            (1, 1, 651, 3),  # 651 1x5 blocks, nblocks_per_core = 6 -> working
            # Anyway, we need to understand how the shapes are mapped to cores,
            # because for these reader/writer kernels, some shapes fail both for WH
            # and BH
            (1, 1, 1, 2),
            (5, 2, 4, 8),
            (5, 2, 4, 7),
            # ## resnet shapes
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

    # If we put use_multicore to False, everything passes
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
    # if first_non_equal_row is not None:
    #     logger.error(f"First non-equal row: {first_non_equal_row}")
    #     logger.error(f"Golden result: {golden_res[first_non_equal_row]}")
    #     logger.error(f"Our result: {our_res[first_non_equal_row]}")

    # ipdb.set_trace()

    assert passing1
