import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from torch.nn import functional as F


@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("scalar", [0.125])
@pytest.mark.parametrize("batch_size", [6, 7, 8])
def test_multiply_with_scalar_sharded(device, scalar, batch_size, output_memory_config):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.rand((batch_size, 16, 384, 384), dtype=torch.float32)
    torch_output_tensor = scalar * torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, device=device
    )
    output = ttnn.mul(input_tensor_a, scalar, memory_config=output_memory_config)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


# E       RuntimeError: TT_THROW @ ../tt_metal/impl/allocator/allocator.cpp:142: tt::exception
# E       info:
# E       Out of Memory: Not enough space to allocate 56623104 B L1 buffer across 56 banks, where each bank needs to store 1011712 B
# E       backtrace:
# E        --- /home/ubuntu/Sabira/tt-metal/build_Release_tracy/lib/libtt_metal.so(+0xedf23) [0x7f6b00ebbf23]
