# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
    infer_ttnn_module_args,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.experimental.ufld_v2_rn18like.reference.ufld_v2_rn18like_model import TuSimple18like, BasicBlock
from models.experimental.ufld_v2_rn18like.tt.ttnn_ufld_v2_rn18like import TtnnUFLDV2RN18like
from models.experimental.ufld_v2_rn18like.tt.ttnn_basic_block import TtnnBasicBlock
from tests.ttnn.utils_for_testing import assert_with_pcc


def custom_preprocessor_basic_block(model, name):
    parameters = {}
    if isinstance(model, BasicBlock):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["conv1"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
        parameters["conv2"] = {}
        parameters["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 64, 80, 200),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ufld_v2_rn18_basic_block(device, batch_size, input_channels, height, width):
    torch_model = TuSimple18like(input_height=height, input_width=width).model.layer1[0]
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width), dtype=torch.bfloat16)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        (ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2]),
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor_basic_block,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnBasicBlock(parameters.conv_args, parameters, device=device)
    torch_out = torch_model(torch_input_tensor)
    ttnn_output = ttnn_model(ttnn_input_tensor)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_out.shape)
    assert_with_pcc(ttnn_output, torch_out, 0.99)


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, TuSimple18like):
        # conv1,bn1
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.conv1, model.model.bn1)
        parameters["model"] = {}
        parameters["model"]["conv1"] = {}
        parameters["model"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # layer0 - 0
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer1[0].conv1, model.model.layer1[0].bn1)
        parameters["model"]["layer1_0"] = {}
        parameters["model"]["layer1_0"]["conv1"] = {}
        parameters["model"]["layer1_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer1_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer1[0].conv2, model.model.layer1[0].bn2)
        parameters["model"]["layer1_0"]["conv2"] = {}
        parameters["model"]["layer1_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer1_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # layer1 - 1
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer1[1].conv1, model.model.layer1[1].bn1)
        parameters["model"]["layer1_1"] = {}
        parameters["model"]["layer1_1"]["conv1"] = {}
        parameters["model"]["layer1_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer1_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer1[1].conv2, model.model.layer1[1].bn2)
        parameters["model"]["layer1_1"]["conv2"] = {}
        parameters["model"]["layer1_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer1_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # layer-2-0
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer2[0].conv1, model.model.layer2[0].bn1)
        parameters["model"]["layer2_0"] = {}
        parameters["model"]["layer2_0"]["conv1"] = {}
        parameters["model"]["layer2_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer2_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer2[0].conv2, model.model.layer2[0].bn2)
        parameters["model"]["layer2_0"]["conv2"] = {}
        parameters["model"]["layer2_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer2_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # layer2 - 1
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer2[1].conv1, model.model.layer2[1].bn1)
        parameters["model"]["layer2_1"] = {}
        parameters["model"]["layer2_1"]["conv1"] = {}
        parameters["model"]["layer2_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer2_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer2[1].conv2, model.model.layer2[1].bn2)
        parameters["model"]["layer2_1"]["conv2"] = {}
        parameters["model"]["layer2_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer2_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # downsample layer2[0]
        if hasattr(model.model.layer2[0], "downsample") and model.model.layer2[0].downsample is not None:
            downsample = model.model.layer2[0].downsample
            if isinstance(downsample, torch.nn.Sequential):
                conv_layer = downsample[0]
                bn_layer = downsample[1]
                weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters["model"]["layer2_0"]["downsample"] = {}
                parameters["model"]["layer2_0"]["downsample"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
                bias = bias.reshape((1, 1, 1, -1))
                parameters["model"]["layer2_0"]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # layer3-0
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer3[0].conv1, model.model.layer3[0].bn1)
        parameters["model"]["layer3_0"] = {}
        parameters["model"]["layer3_0"]["conv1"] = {}
        parameters["model"]["layer3_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer3_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer3[0].conv2, model.model.layer3[0].bn2)
        parameters["model"]["layer3_0"]["conv2"] = {}
        parameters["model"]["layer3_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer3_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # layer3-1
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer3[1].conv1, model.model.layer3[1].bn1)
        parameters["model"]["layer3_1"] = {}
        parameters["model"]["layer3_1"]["conv1"] = {}
        parameters["model"]["layer3_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer3_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer3[1].conv2, model.model.layer3[1].bn2)
        parameters["model"]["layer3_1"]["conv2"] = {}
        parameters["model"]["layer3_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer3_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # downsample - layer3[0]
        if hasattr(model.model.layer3[0], "downsample") and model.model.layer3[0].downsample is not None:
            downsample = model.model.layer3[0].downsample
            if isinstance(downsample, torch.nn.Sequential):
                conv_layer = downsample[0]
                bn_layer = downsample[1]
                weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters["model"]["layer3_0"]["downsample"] = {}
                parameters["model"]["layer3_0"]["downsample"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
                bias = bias.reshape((1, 1, 1, -1))
                parameters["model"]["layer3_0"]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # layer4-0
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer4[0].conv1, model.model.layer4[0].bn1)
        parameters["model"]["layer4_0"] = {}
        parameters["model"]["layer4_0"]["conv1"] = {}
        parameters["model"]["layer4_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer4_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer4[0].conv2, model.model.layer4[0].bn2)
        parameters["model"]["layer4_0"]["conv2"] = {}
        parameters["model"]["layer4_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer4_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # layer4 - 1
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer4[1].conv1, model.model.layer4[1].bn1)
        parameters["model"]["layer4_1"] = {}
        parameters["model"]["layer4_1"]["conv1"] = {}
        parameters["model"]["layer4_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer4_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.model.layer4[1].conv2, model.model.layer4[1].bn2)
        parameters["model"]["layer4_1"]["conv2"] = {}
        parameters["model"]["layer4_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["model"]["layer4_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # downsample - layer4[0]
        if hasattr(model.model.layer4[0], "downsample") and model.model.layer4[0].downsample is not None:
            downsample = model.model.layer4[0].downsample
            if isinstance(downsample, torch.nn.Sequential):
                conv_layer = downsample[0]
                bn_layer = downsample[1]
                weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters["model"]["layer4_0"]["downsample"] = {}
                parameters["model"]["layer4_0"]["downsample"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
                bias = bias.reshape((1, 1, 1, -1))
                parameters["model"]["layer4_0"]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # pool
        parameters["pool"] = {}
        parameters["pool"]["weight"] = ttnn.from_torch(model.pool.weight, dtype=ttnn.float32)
        if model.pool.bias is not None:
            bias = model.pool.bias.reshape((1, 1, 1, -1))
            parameters["pool"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        else:
            parameters["pool"]["bias"] = None
        parameters["cls"] = {}
        parameters["cls"]["conv_1"] = {}
        parameters["cls"]["conv_1"]["weight"] = ttnn.from_torch(model.cls[1].weight, dtype=ttnn.float32)
        if model.cls[1].bias is not None:
            bias = model.cls[1].bias.reshape((1, 1, 1, -1))
            parameters["cls"]["conv_1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        else:
            parameters["cls"]["conv_1"]["bias"] = None

        parameters["cls"]["linear_1"] = {}
        parameters["cls"]["linear_1"]["weight"] = preprocess_linear_weight(model.cls[4].weight, dtype=ttnn.bfloat16)
        if model.cls[4].bias is not None:
            parameters["cls"]["linear_1"]["bias"] = preprocess_linear_bias(model.cls[4].bias, dtype=ttnn.bfloat16)
        else:
            parameters["cls"]["linear_1"]["bias"] = None

    return parameters


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 320, 800),
        (2, 3, 320, 800),
        (4, 3, 320, 800),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        # True
    ],  # uncomment  to run the model for real weights
    ids=[
        "pretrained_weight_false",
        # "pretrained_weight_true",  # uncomment to run the model for real weights
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ufld_rn18like(device, batch_size, input_channels, height, width, use_pretrained_weight):
    torch_model = TuSimple18like(input_height=height, input_width=width)
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width), dtype=torch.bfloat16)
    if use_pretrained_weight:
        wts = torch.load("models/experimental/ufld_v2_rn18like/ufldv2_resnet18_trained_statedict.pth")
        torch_model.load_state_dict(wts)
    torch_output = torch_model(torch_input_tensor)
    n, c, h, w = torch_input_tensor.shape
    if c == 3:
        c = 16
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT  # , device=device
    )
    ttnn_input_tensor = ttnn_input_tensor.to(device, input_mem_config)
    # ttnn_input_tensor pass sharded tensor and check
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnUFLDV2RN18like(conv_args=parameters.conv_args, conv_pth=parameters, device=device)
    torch_output, pred_list = torch_model(torch_input_tensor)
    ttnn_output = ttnn_model(input=ttnn_input_tensor, batch_size=batch_size)
    ttnn_output = ttnn.to_torch(ttnn_output).squeeze(dim=0).squeeze(dim=0)
    assert_with_pcc(ttnn_output, torch_output, 0.99)
