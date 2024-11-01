import pytest
import ttnn
from loguru import logger

from models.utility_functions import comp_pcc
import torch

def test_attn(device):
    torch.manual_seed(0)
    at = torch.ones([1,32,32,32]) * 1
    bt = torch.ones([32,1,32,32]) * 1
    a=ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    b=ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    compute_grid_size = device.compute_with_storage_grid_size()
    c = ttnn.experimental.attn_matmul(a,b, compute_with_storage_grid_size=compute_grid_size,dtype=ttnn.bfloat8_b)
    #print(c)
    golden_output_tensor = (at.transpose(0, 2) @ bt).transpose(0, 2)
    #print(f'go {golden_output_tensor}')
    tt_output_tensor = c.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    #print(f'tt {tt_output_tensor}')
    allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
    assert allclose, f"FAILED: {output}"