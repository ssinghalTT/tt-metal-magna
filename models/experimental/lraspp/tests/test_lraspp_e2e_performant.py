# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time
import torch
from loguru import logger

from models.utility_functions import run_for_wormhole_b0
from models.experimental.lraspp.tests.lraspp_e2e_performant import LRASPPTrace2CQ


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 8],
)
def test_run_lraspp_trace_2cq_inference(
    device,
    use_program_cache,
    batch_size,
    model_location_generator,
):
    lraspp_trac2_2cq = LRASPPTrace2CQ()

    lraspp_trac2_2cq.initialize_lraspp_trace_2cqs_inference(
        device,
        batch_size,
        model_location_generator=None,
    )

    input_shape = (batch_size, 3, 224, 224)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    n, c, h, w = torch_input_tensor.shape
    torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
    torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
    # tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    # tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
    inference_iter_count = 100
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        t0 = time.time()
        tt_inputs_host = torch.nn.functional.pad(torch_input_tensor, (0, 13), "constant", 0)
        tt_inputs_host = ttnn.from_torch(tt_inputs_host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        output = lraspp_trac2_2cq.execute_lraspp_trace_2cqs_inference(tt_inputs_host)
        t1 = time.time()
        inference_time_iter.append(t1 - t0)
    lraspp_trac2_2cq.release_lraspp_trace_2cqs_inference()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    logger.info(
        f"ttnn_lraspp_224x224_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
