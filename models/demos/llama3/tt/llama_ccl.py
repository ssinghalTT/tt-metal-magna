# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def tt_all_reduce(
    input_tensor,
    mesh_device,
    cluster_axis=0,
    dim=0,
    num_links=2,
    memory_config=None,
    sharded=False,
    dtype=ttnn.bfloat16,
):
    # N150
    if mesh_device.shape == (1, 1) or (cluster_axis == 1 and 1 in mesh_device.shape):
        return input_tensor

    if input_tensor.dtype != dtype:
        if input_tensor.is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)
        input_tensor = ttnn.typecast(input_tensor, dtype)

    # Ensure dim 0 and 1 are 1
    original_shape = input_tensor.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        input_tensor = ttnn.reshape(
            input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    # N300 and T3K
    if 1 in mesh_device.shape:  # TODO: use reducescatter and all gather for all models
        if input_tensor.is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)
        return ttnn.reduce_scatter(
            input_tensor,
            scatter_dim=dim,
            math_op=ttnn.ReduceType.Sum,
            num_links=num_links,
            memory_config=memory_config,
        )
    # TG

    # Ensure the input tensor is in the correct memory configuration
    if not sharded:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
    gathered_tensor = ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
    )

    # print(f"{gathered_tensor}")
    if sharded:
        gathered_tensor = ttnn.to_memory_config(gathered_tensor, ttnn.L1_MEMORY_CONFIG)

    reduced_tensors = ttnn.experimental.fast_reduce_nc(
        gathered_tensor,
        dims=[dim],
        output=None,
        compute_kernel_config=None,
        memory_config=ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG,
    )

    # Reshape the reduced tensor to the original shape
    reduced_tensors = ttnn.reshape(reduced_tensors, original_shape)

    return reduced_tensors


def tt_all_gather(
    input_tensor,
    mesh_device,
    cluster_axis,
    dim,
    num_links=2,
    memory_config=None,
    sharded=False,
    topology=ttnn.Topology.Linear,
    dtype=ttnn.bfloat16,
):
    # N150
    if mesh_device.shape == (1, 1) or (cluster_axis == 1 and 1 in mesh_device.shape):
        return input_tensor

    # Ensure the input tensor is in the correct memory configuration
    if not sharded:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    if input_tensor.dtype != dtype:
        if input_tensor.is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)
        input_tensor = ttnn.typecast(input_tensor, dtype)

    if cluster_axis is None:
        return ttnn.all_gather(
            input_tensor,
            dim,
            num_links=num_links,
            # mesh_device=mesh_device,
            topology=topology,
            memory_config=memory_config,
        )

    return ttnn.all_gather(
        input_tensor,
        dim,
        num_links=num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=topology,
        memory_config=memory_config,
    )


def tt_distributed_rmsnorm(inp, epsilon, gamma, mesh_device, compute_kernel_config):
    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)

    padded_shape = (1, 1, inp.shape[-2], 32)
    tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape, padded_shape))  # TODO: Figure out why we need this
    tt_stats = tt_all_gather(
        tt_stats,
        mesh_device=mesh_device,
        dim=3,
        cluster_axis=1,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats, epsilon=epsilon, weight=gamma, compute_kernel_config=compute_kernel_config
    )

    tt_stats.deallocate(True)

    return tt_out


def tt_sharded_distributed_rmsnorm(
    inp, epsilon, gamma, mesh_device, ln_sharded_input_memcfg, ln_sharded_progcfg, ln_sharded_stats_memcfg
):
    inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, program_config=ln_sharded_progcfg)

    # All gather stats
    tt_stats = ttnn.all_gather(
        tt_stats,
        3,
        num_links=1,
        cluster_axis=1,
        mesh_device=mesh_device,
        memory_config=ln_sharded_stats_memcfg,
        topology=ttnn.Topology.Linear,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp,
        epsilon=epsilon,
        weight=gamma,
        program_config=ln_sharded_progcfg,
        stats=tt_stats,
    )
    tt_stats.deallocate(True)

    return tt_out
