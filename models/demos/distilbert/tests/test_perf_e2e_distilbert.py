# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.utility_functions import run_for_wormhole_b0

from models.demos.distilbert.tests.perf_e2e_distilbert import run_perf_distilbert


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1797120}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((8, 15, 16),),
)
def test_perf_trace_2cqs(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
):
    print("Starting")
    run_perf_distilbert(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        device,
        "distilbert-base-uncased-distilled-squad",
    )
