# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import time
import torch
import pytest
from loguru import logger
from models.utility_functions import is_wormhole_b0
from models.perf.perf_utils import prep_perf_report
from models.experimental.ufld_v2_rn18like.ttnn.ttnn_ufld_v2_rn18like import (
    TtnnUFLDV2RN18like,
)
from models.experimental.ufld_v2_rn18like.reference.ufld_v2_rn18like_model import TuSimple18like
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.experimental.ufld_v2_rn18like.demo.demo import custom_preprocessor_whole_model


def get_expected_times(name):
    base = {"ufld_v2_rn18like": (25.1, 0.28)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (4, 3, 320, 800),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        #  True
    ],  # uncomment  to run the model for real weights
    ids=[
        "pretrained_weight_false",
        # "pretrained_weight_true",  # uncomment to run the model for real weights
    ],
)
def test_ufld_v2_rn18like_model_perf(device, batch_size, input_channels, height, width, use_pretrained_weight):
    disable_persistent_kernel_cache()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width))
    reference_model = TuSimple18like(input_height=height, input_width=width)
    if use_pretrained_weight:
        wts = torch.load("models/experimental/ufld_rn18like/demo/ufldv2_resnet18_trained_statedict.pth")
        reference_model.load_state_dict(wts)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor_whole_model,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=reference_model, run_model=lambda model: reference_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnUFLDV2RN18like(conv_args=parameters.conv_args, conv_pth=parameters, device=device)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        (ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2]),
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    durations = []

    for i in range(2):
        start = time.time()
        ttnn_model_output_1 = ttnn_model(ttnn_input_tensor, batch_size=batch_size)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output_1)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("ufld_v2_rn18like")

    prep_perf_report(
        model_name="models/experimental/ufld_v2_rn18like",
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
    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"


@pytest.mark.parametrize(
    "batch_size, expected_perf,test",
    [
        [4, 325, "ufld_v2_rn18like"],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_ufld_v2_rn18like(batch_size, expected_perf, test):
    subdir = "ttnn_ufld_v2_rn18like"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest models/experimental/ufld_v2_rn18like/demo/demo.py::test_ufld_v2_res18like_inference"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_ufld_v2_rn18like{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
