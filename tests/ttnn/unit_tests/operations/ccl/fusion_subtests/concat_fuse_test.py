# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tracy import signpost


def run_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor_mesh,
    buffer_tensor_mesh,
    dim,
    num_links,
    output_mem_config,
    enable_persistent_fabric,
    multi_device_global_semaphore,
    num_iter=20,
    subdevice_id=None,
    profiler=BenchmarkProfiler(),
    warmup_iters=0,
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_gather_concat(
        input_tensor_mesh,
        buffer_tensor_mesh,
        dim,
        mesh_device=mesh_device,
        cluster_axis=1,
        multi_device_global_semaphore=multi_device_global_semaphore,
        num_links=num_links,
        num_heads=8,
        memory_config=output_mem_config,
        topology=all_gather_topology,
        subdevice_id=subdevice_id,
        enable_persistent_fabric_mode=enable_persistent_fabric,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")

    if warmup_iters > 0:
        trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(warmup_iters):
            tt_out_tensor = ttnn.experimental.all_gather_concat(
                input_tensor_mesh,
                buffer_tensor_mesh,
                dim,
                mesh_device=mesh_device,
                cluster_axis=1,
                multi_device_global_semaphore=multi_device_global_semaphore,
                num_links=num_links,
                num_heads=8,
                memory_config=output_mem_config,
                topology=all_gather_topology,
                subdevice_id=subdevice_id,
                enable_persistent_fabric_mode=enable_persistent_fabric,
            )
        ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
        ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.all_gather_concat(
            input_tensor_mesh,
            buffer_tensor_mesh,
            dim,
            mesh_device=mesh_device,
            cluster_axis=1,
            multi_device_global_semaphore=multi_device_global_semaphore,
            num_links=num_links,
            num_heads=8,
            memory_config=output_mem_config,
            topology=all_gather_topology,
            subdevice_id=subdevice_id,
            enable_persistent_fabric_mode=enable_persistent_fabric,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")

    profiler.start("all-gather-concat-trace-warmup")
    if warmup_iters > 0:
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        ttnn.synchronize_device(mesh_device)
    profiler.end("all-gather-concat-trace-warmup")

    signpost("start")
    profiler.start("all-gather-concat-trace")

    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    profiler.end("all-gather-concat-trace")
    signpost("stop")
    time_taken = profiler.get_duration("all-gather-concat-trace") - profiler.get_duration(
        "all-gather-concat-trace-warmup"
    )

    return tt_out_tensor


def run_concat_fuse_impl(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    all_gather_topology,
    num_iters=1,
    enable_async=False,
    trace_mode=False,
    output_shard_shape=None,
    output_shard_grid=None,
    tensor_mem_layout=None,
    warmup_iters=0,
    profiler=BenchmarkProfiler(),
):
    enable_persistent_fabric = True
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
        mesh_device,
        [worker_sub_device],
        0,
        0,
        enable_persistent_fabric,
        wrap_fabric_around_mesh=True,
    )
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ### For sharded all gather only
    if bool(input_shard_shape) != bool(input_shard_grid) and bool(tensor_mem_layout) != bool(input_shard_grid):
        pytest.fail(
            "Both input_shard_shape, shard_grid, and tensor_mem_layout must be provided together or all must be None"
        )
    if input_shard_shape and input_shard_grid:
        input_shard_spec = ttnn.ShardSpec(
            input_shard_grid,
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec
        )
        if output_shard_shape is None:
            assert (
                output_shard_grid is None
            ), "output_shard_grid must not be provided if output_shard_shape is not provided"
            output_shard_shape = list(input_shard_shape)
            if dim == len(output_shape) - 1:
                output_shard_shape[1] *= num_devices
            else:
                output_shard_shape[0] *= num_devices
            output_shard_spec = ttnn.ShardSpec(
                input_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
        else:
            assert output_shard_grid is not None, "output_shard_grid must be provided if output_shard_shape is provided"
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
    ###

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []
    intermediate_tensors_list = []

    for i in range(num_iters):
        output_tensor = torch.rand(output_shape).bfloat16()
        output_tensor_goldens_list.append(output_tensor)
        input_tensors = torch.chunk(output_tensor, num_devices, dim)
        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(
                ttnn.Tensor(t, input_dtype).to(layout).to(mesh_device.get_devices()[i], input_mem_config)
            )
            logger.info(f"using device {mesh_device.get_devices()[i].id()}")

        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

        input_tensor_mesh_list.append(input_tensor_mesh)

        intermediate_core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 0)),
            }
        )
        intermediate_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                intermediate_core_range_set,
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        zeros_tensor = torch.zeros(output_shape).bfloat16()
        tt_intermediate_tensors = []
        for i in range(4):
            tt_intermediate_tensor = (
                ttnn.Tensor(zeros_tensor, input_dtype)
                .to(layout)
                .to(mesh_device.get_devices()[i], intermediate_mem_config)
            )
            tt_intermediate_tensors.append(tt_intermediate_tensor)
        intermediate_tensor_mesh = ttnn.aggregate_as_tensor(tt_intermediate_tensors)
        intermediate_tensors_list.append(intermediate_tensor_mesh)
    tt_out_tensor_list = []

    if trace_mode:
        tt_out_tensor = run_with_trace(
            mesh_device,
            all_gather_topology,
            input_tensor_mesh_list[0],
            intermediate_tensors_list[0],
            dim,
            num_links,
            output_mem_config,
            enable_persistent_fabric,
            multi_device_global_semaphore=ccl_semaphore_handles[0],
            num_iter=num_iters,
            subdevice_id=worker_sub_device_id,
            profiler=profiler,
            warmup_iters=warmup_iters,
        )
        tt_out_tensor_list.append(tt_out_tensor)
    else:
        for i in range(num_iters):
            tt_out_tensor = ttnn.experimental.all_gather_concat(
                input_tensor_mesh_list[i],
                intermediate_tensors_list[i],
                dim,
                cluster_axis=1,
                mesh_device=mesh_device,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                num_heads=8,
                memory_config=output_mem_config,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
                enable_persistent_fabric_mode=enable_persistent_fabric,
            )
            tt_out_tensor_list.append(tt_out_tensor)

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    passed = True
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_out_tensor = tt_out_tensor_list[tensor_index]
        output_tensor = output_tensor_goldens_list[tensor_index]
        output_tensor = output_tensor[:, :, :8, :]
        output_tensor_concat = output_tensor.reshape(1, 1, 32, 1024)
        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking for device {t.device().id()}")

            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(tt_output_tensor, output_tensor_concat)
            else:
                eq, output = comp_pcc(tt_output_tensor, output_tensor_concat)
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
                passed = False

    mesh_device.reset_sub_device_stall_group()
    teardown_fabric_interface(mesh_device)
    print("teardown done\n")

    if not passed:
        assert eq, f"{i} FAILED: {output}"
