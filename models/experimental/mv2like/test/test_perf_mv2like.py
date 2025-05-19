# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
import ttnn
from loguru import logger
from models.experimental.mv2like.reference.mv2_like import Mv2Like
from models.experimental.mv2like.tt.ttnn_mv2_like import TtMv2Like
from models.utility_functions import (
    disable_persistent_kernel_cache,
)
from models.experimental.mv2like.tt.model_preprocessing import (
    create_mv2_like_input_tensors,
    create_mv2_like_model_parameters,
)
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import (
    profiler,
)


def get_expected_times(name):
    base = {"mv2like": (63.3, 0.14)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 8],
)
def test_mv2like(device, batch_size, reset_seeds):
    disable_persistent_kernel_cache()
    profiler.clear()
    weights_path = "models/experimental/mv2like/lraspp_mobilenet_v2_trained_statedict.pth"

    state_dict = torch.load(weights_path)
    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = Mv2Like()
    new_state_dict = {
        name1: parameter2
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
        if isinstance(parameter2, torch.FloatTensor)
    }
    torch_model.load_state_dict(new_state_dict)

    torch_model.eval()

    model_parameters = create_mv2_like_model_parameters(torch_model, device=device)
    torch_input_tensor, ttnn_input_tensor = create_mv2_like_input_tensors(
        batch=batch_size, input_height=224, input_width=224
    )
    n, c, h, w = torch_input_tensor.shape
    if c == 3:
        c = 16
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=7),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input_tensor = ttnn_input_tensor.to(device, input_mem_config)

    ttnn_model = TtMv2Like(model_parameters, device, batchsize=batch_size)

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    ttnn_output_tensor = ttnn_model(ttnn_input_tensor)

    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    iterations = 16
    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        ttnn.deallocate(ttnn_output_tensor)
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        ttnn_output_tensor = ttnn_model(ttnn_input_tensor)
        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation took {compile_time:.1f} s")
    logger.info(f"Inference time on last iterations was completed in {(inference_time * 1000.0):.2f} ms")
    logger.info(
        f"Mean inference time for {batch_size} (batch) images was {(mean_inference_time * 1000.0):.2f} ms ({batch_size / mean_inference_time:.2f} fps)"
    )

    expected_compile_time, expected_inference_time = get_expected_times("mv2like")

    prep_perf_report(
        model_name="models/experimental/Mv2Like",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 93],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_mv2like(batch_size, expected_perf):
    subdir = "ttnn_mv2like"
    num_iterations = 1
    margin = 0.03
    device_params_str = "{'l1_small_size': 32768}"
    command = f"pytest models/experimental/mv2like/test/test_ttnn_mv2_like.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_mv2like{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
