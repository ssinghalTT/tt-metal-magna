# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import FeedForward
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (32,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
def test_llama_mlp_inference(mesh_device, seq_len, use_program_cache, reset_seeds, ensure_gc):
    dim_in = 4096
    dim_hidden = int(3.5 * dim_in / 4)  # 3584
    dim_out = dim_in

    # Create random input tensor
    input_tensor = torch.randn(1, 1, int(seq_len), dim_in)

    # Create random weight matrices
    w1 = torch.randn(dim_hidden, dim_in)
    w2 = torch.randn(dim_out, dim_hidden)

    # Pytorch reference implementation
    ## First linear layer
    hidden = torch.matmul(input_tensor, w1.t())
    ## Second linear layer
    output_w2 = torch.matmul(hidden, w2.t())
    ## Add residual connection
    reference_output = output_w2 + input_tensor

    # TTNN implementation
    ## memory and compute kernel config
    # args = TtModelArgs(mesh_device, instruct=False, dummy_weights=True)
    # input_mem_config = args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
    # w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
    # w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)
    # pc_1 = args.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
    # pc_2 = args.model_config["DECODE_MLP_W2_PRG_CONFIG"]
    # print(f"input_mem_config: {input_mem_config}")
    # print(f"w1_w3_mem_config: {w1_w3_mem_config}")
    # print(f"w2_mem_config: {w2_mem_config}")
    # print(f"pc_1: {pc_1}")
    # print(f"pc_2: {pc_2}")
    # breakpoint()

    input_mem_config = ttnn.create_sharded_memory_config(
        (
            32,
            128,
        ),  # Shard shape: [32, 128] -> 1 shard per core
        ttnn.CoreGrid(x=8, y=4),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    w1_out_reshard_mem_config = ttnn.create_sharded_memory_config(
        (
            32,
            128,
        ),  # Shard shape: [32, 128] -> 1 shard per core
        ttnn.CoreGrid(x=7, y=4),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    dram_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(11, 0),
            ),
        }
    )
    w1_w3_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, (4096, 320), ttnn.ShardOrientation.ROW_MAJOR, False),
    )
    w2_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, (3584, 352), ttnn.ShardOrientation.ROW_MAJOR, False),
    )
    pc_1 = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=4,
        per_core_M=1,
        per_core_N=4,
        fused_activation=None,
    )
    pc_2 = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=4,
        per_core_M=1,
        per_core_N=5,
        fused_activation=None,
    )
    # print(f"input_mem_config: {input_mem_config}")
    # print(f"w1_w3_mem_config: {w1_w3_mem_config}")
    # print(f"w2_mem_config: {w2_mem_config}")
    # print(f"pc_1: {pc_1}")
    # print(f"pc_2: {pc_2}")

    ## convert input tensor and weights to TTNN tensors
    tt_input = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )
    as_sharded_tensor = lambda w, type, dim, mem_config: ttnn.as_tensor(
        w,  # Grab only the wX part of the name
        dtype=type,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
    )
    # Sharded weights
    tt_w1 = as_sharded_tensor(w1.t(), ttnn.bfloat8_b, dim=-1, mem_config=w1_w3_mem_config)
    tt_w2 = as_sharded_tensor(w2.t(), ttnn.bfloat8_b, dim=-2, mem_config=w2_mem_config)

    ## MLP takes replicated inputs and produces fractured outputs
    w1_out = ttnn.linear(
        tt_input,
        tt_w1,
        core_grid=None,
        dtype=ttnn.bfloat16,
        program_config=pc_1,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    logger.info(f"w1_out shape: {w1_out.shape}")
    logger.info(f"w1_out memory config: {w1_out.memory_config()}")
    logger.warning("w1_out shape does not match memory config!")
    w1_out = ttnn.reshard(w1_out, w1_out_reshard_mem_config)
    w2_out = ttnn.linear(
        w1_out,
        tt_w2,
        core_grid=None,
        dtype=ttnn.bfloat16,
        program_config=pc_2,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    w2_out = ttnn.sharded_to_interleaved(w2_out, ttnn.L1_MEMORY_CONFIG)
    tt_input = ttnn.sharded_to_interleaved(tt_input, ttnn.L1_MEMORY_CONFIG)
    ## Add residual connection
    tt_w2_out_torch = ttnn.to_torch(w2_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_input_torch = ttnn.to_torch(tt_input, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_output = ttnn.add(tt_input, w2_out)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    pcc_required = 0.99
    passing_w2_out, pcc_message_w2_out = comp_pcc(output_w2, tt_w2_out_torch)
    passing_input, pcc_message_input = comp_pcc(input_tensor, tt_input_torch)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"w2_out PCC: {pcc_message_w2_out}")
    logger.info(f"input PCC: {pcc_message_input}")
    logger.info(f"residual PCC: {pcc_message}")
