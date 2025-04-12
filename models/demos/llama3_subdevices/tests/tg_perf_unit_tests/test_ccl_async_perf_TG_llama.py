# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

THRESHOLD = 0.4


@pytest.mark.parametrize(
    "ag_type, warmup_iters, perf_target_us",
    [
        ("sdpa", 15, 12.9),
        ("binary_mult", 15, 12.54),
        ("layernorm", 15, 4.87),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ag_tg_llama_perf(
    ag_type,
    warmup_iters,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather_{ag_type}"

    subdir = "llama_ccl_perf"
    command = (
        f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_gather_tg_llama -k {ag_type}"
    )
    cols = ["DEVICE KERNEL"]
    op_name = "AllGatherAsync"
    warmup_iters = warmup_iters * 32  # 5 iterations per device

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min_us = results[cols[0]]["MIN"] / 1000
    measured_max_us = results[cols[0]]["MAX"] / 1000
    measured_avg_us = results[cols[0]]["AVG"] / 1000
    measured_std_us = results[cols[0]]["STD"] / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_gather-{ag_type}-min-us", measured_min_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_gather-{ag_type}-max-us", measured_max_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_gather-{ag_type}-avg-us", measured_avg_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_gather-{ag_type}-std-us", measured_std_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"all_gather",
        ml_model_name="llama70b-tg-ccl",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "ar_type, warmup_iters, perf_target_us",
    [
        ("ff2", 15, 18.6),
        ("qkv", 15, 11.9),
        ("ff1", 15, 19.2),
        ("lm_head", 15, 61.8),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ar_tg_llama_perf(
    ar_type,
    warmup_iters,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_reduce_{ar_type}"

    subdir = "llama_ccl_perf"
    command = (
        f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_reduce_tg_llama -k {ar_type}"
    )
    cols = ["DEVICE KERNEL"]
    op_name = "AllReduceAsync"
    warmup_iters = warmup_iters * 32  # 5 iterations per device

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min_us = results[cols[0]]["MIN"] / 1000
    measured_max_us = results[cols[0]]["MAX"] / 1000
    measured_avg_us = results[cols[0]]["AVG"] / 1000
    measured_std_us = results[cols[0]]["STD"] / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_reduce-{ar_type}-min-us", measured_min_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_reduce-{ar_type}-max-us", measured_max_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_reduce-{ar_type}-avg-us", measured_avg_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_reduce-{ar_type}-std-us", measured_std_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"all_reduce",
        ml_model_name="llama70b-tg-ccl",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "ar_type, warmup_iters, perf_target_us",
    [
        ("rms", 10, 25),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_rms_perf(
    ar_type,
    warmup_iters,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"rms_test"

    subdir = "llama_ccl_perf"
    command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_tg_trace_rms_fuse"
    cols = ["DEVICE KERNEL"]
    op_name = "RMSAllGather"
    warmup_iters = warmup_iters * 32  # 5 iterations per device

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=0)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min_us = results[cols[0]]["MIN"] / 1000
    measured_max_us = results[cols[0]]["MAX"] / 1000
    measured_avg_us = results[cols[0]]["AVG"] / 1000
    measured_std_us = results[cols[0]]["STD"] / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"rms-min-us", measured_min_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"rms-max-us", measured_max_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"rms-avg-us", measured_avg_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"rms-std-us", measured_std_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"rms_test",
        ml_model_name="llama70b-tg-ccl",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "warmup_iters, perf_target_us",
    [
        (5, 17),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_fused_all_gather_concat_perf(
    warmup_iters,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather_concat_heads"

    subdir = "llama_ccl_perf"
    command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_concat_fuse"
    cols = ["DEVICE KERNEL"]
    op_name = "AllGatherConcat"
    warmup_iters = warmup_iters * 32  # 5 iterations per device

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=0)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min_us = results[cols[0]]["MIN"] / 1000
    measured_max_us = results[cols[0]]["MAX"] / 1000
    measured_avg_us = results[cols[0]]["AVG"] / 1000
    measured_std_us = results[cols[0]]["STD"] / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_gather-concat_heads-min-us", measured_min_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_gather-concat_heads-max-us", measured_max_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_gather-concat_heads-avg-us", measured_avg_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"all_gather-concat_heads-std-us", measured_std_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"all_gather_concat_heads_fused",
        ml_model_name="llama70b-tg-ccl",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
