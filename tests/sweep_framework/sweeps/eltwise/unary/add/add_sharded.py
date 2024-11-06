# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
import math
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape_rm, get_device_grid_size
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30
Y, X = get_device_grid_size()

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 256, 256], [6, 12, 512, 512], [1, 1, 256, 256], 4)
        + gen_shapes([1, 256, 256], [12, 512, 512], [1, 256, 256], 4)
        + gen_shapes([256, 256], [512, 512], [256, 256], 4),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "sharding_strategy": [ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
        "shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "xfail": {
        "input_shape": gen_shapes([1, 1, 96, 96], [6, 12, 512, 512], [1, 1, 16, 16], 4)
        + gen_shapes([1, 96, 96], [12, 512, 512], [1, 16, 16], 4)
        + gen_shapes([96, 96], [512, 512], [16, 16], 4),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "sharding_strategy": [ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
        "shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "test": {
        "input_shape": [[1, 1, 260, 260]],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "sharding_strategy": [ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
        "shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if math.prod(test_vector["input_shape"][:-1]) % Y != 0:
        return True, "Prod of all dimensions except the innermost must be divisible by the y coordinate of coregrid"
    if test_vector["input_shape"][-1] % X != 0:
        return True, "Innermost dimension must be divisible by the x coordinate of coregrid"
    if math.prod(test_vector["input_shape"][:-1]) < (Y * 32):
        return (
            True,
            "Prod of all dimensions except the innermost must be greater or equal to y coordinate of coregrid times 32",
        )
    if test_vector["input_shape"][-1] < (X * 32):
        return True, "Innermost dimension must be greater or equal to x coordinate of coregrid times 32"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Row Major layout on input tensor is not supported when sharding it"

    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_layout,
    sharding_strategy,
    shard_orientation,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    scalar = torch.tensor(1, dtype=torch.bfloat16).uniform_(-100, 100).item()

    golden_function = ttnn.get_golden_function(ttnn.add)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    sharded_config = ttnn.create_sharded_memory_config(
        shape=input_shape,
        core_grid=device_grid_size,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )

    input_tensor_a = ttnn.to_memory_config(input_tensor_a, sharded_config)

    start_time = start_measuring_time()
    output_tensor = ttnn.add(input_tensor_a, scalar)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
