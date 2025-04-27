# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, WS, BS
import ttnn
import torch

from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (353, 384, 8, 8, WS, None),
        (128, 128, 32, 32, BS, None),
        (16, 16, 256, 256, HS, {"act_block_h": 32}),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [True, False],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [False],
)
@pytest.mark.parametrize(
    "filter, padding",
    [
        [3, (1, 2, 2, 3)],
        [1, 0],
        [5, (2, 4, 3, 5)],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv_features(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    padding,
    output_layout,
    fp32_accum,
    packer_l1_acc,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and shard_layout == WS:
        pytest.skip("Bug in Width Sharded Row Major Tensor Creation when height%32!=0. #19408")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat16 and packer_l1_acc and fp32_accum:
        pytest.skip("skipping due to pack_untilize_dst issue!")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=True,
        run_twice=True,
    )


SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, slice_type, num_slices, weights_dtype, activations_dtype, kernel, stride, padding, dilation, input_channels_alignment, act_block_h_override,  math_fidelity",
    # fmt: off
    (
        (2, 512,  512,  128,   128,   SliceWidth,    4,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 32 * 8,  ttnn.MathFidelity.LoFi  ),
        (2, 64,   64,   384,   64,    SliceHeight,   6,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (1, 4,    32,   1024,  1024,  SliceWidth,    4,  ttnn.bfloat8_b, ttnn.bfloat16, (5, 5), (1, 1), (0, 0), (1, 1), 16, 32,      ttnn.MathFidelity.LoFi  ),
        (1, 64,   128,  992,   992,   SliceWidth,   64,  ttnn.bfloat8_b, ttnn.bfloat16, (2, 2), (1, 1), (0, 0), (1, 1), 32, 32 * 4,  ttnn.MathFidelity.LoFi  ),
    )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, False, False]],
)
def test_conv_dram(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    activations_dtype,
    slice_type,
    num_slices,
    kernel,
    stride,
    padding,
    dilation,
    input_channels_alignment,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        config,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=False,
        transpose_shards=True,
        run_twice=False,
        fast_compare=False,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        ),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv_sharded_non_tile(device):
    # conv1 input shape Shape([1, 1, 42240, 64])
    # conv1 input mem cfg MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED,buffer_type=BufferType::L1,shard_spec=ShardSpec(
    # grid={[(x=0,y=0) - (x=7,y=6)], [(x=0,y=7) - (x=6,y=7)]},shape={671, 64},
    # orientation=ShardOrientation::ROW_MAJOR,mode=ShardMode::PHYSICAL,physical_shard_shape=std::nullopt))
    # conv1 input layout Layout.ROW_MAJOR
    # conv1 input dtype DataType.BFLOAT16
    # batch size: 1
    # input_height: 528
    # input_width: 80
    # in_channels: 64
    # out_channels: 64
    # kernel_size: (3, 3)
    # padding: (1, 1)
    # stride: (1, 1)
    # groups: 4
    batch = 1
    input_channels = 32
    output_channels = 32
    input_height = 528
    input_width = 80
    # 528*40 = 21120
    # 22120 / 632 = 336
    filter = 3
    stride = 1
    padding = 0
    shard_height = 671  # 671
    shard_width = 32
    input_shape = (batch, input_channels, input_height, input_width)
    weights_shape = (output_channels, input_channels, filter, filter)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    torch_input_nhwc = torch.permute(torch_input, (0, 2, 3, 1))

    input_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
            ]
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # input_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16, device=device, memory_config=input_mem_cfg)
    print(f"tt_input shape: {tt_input.shape}")
    print(f"tt_input mem cfg: {tt_input.memory_config()}")

    torch_weights = torch.randn(weights_shape, dtype=torch.bfloat16)
    tt_weights = ttnn.from_torch(torch_weights, dtype=ttnn.bfloat16)
    [tt_out, [oh, ow]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weights,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        kernel_size=(filter, filter),
        stride=(stride, stride),
        padding=(padding, padding),
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        return_output_dim=True,
    )

    torch_output_tensor = ttnn.to_torch(tt_out)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch, oh, ow, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input, torch_weights, bias=None, stride=stride, padding=padding, groups=1
    )
    print(torch_output_tensor)
    print(torch_out_golden_tensor)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.99)
    logger.info(f"PCC = {pcc_msg}. Threshold = 0.99")
    allclose = torch.allclose(torch_output_tensor, torch_out_golden_tensor, atol=1e-1, rtol=1e-1)
    # assert allclose
    assert passing
