# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import statistics

import torch
import transformers
from loguru import logger
import torch.nn as nn

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.demos.deit_tiny.tt import ttnn_optimized_sharded_deit_wh
from models.utility_functions import torch2tt_tensor, is_blackhole
from models.demos.wormhole.deit_tiny.demo.deit_helper_funcs import (
    get_batch_cifar,
    get_cifar10_label_dict,
    get_data_loader_cifar10,
)

from tests.ttnn.utils_for_testing import assert_with_pcc

import ast


def get_imagenet_label_dict():
    path = "models/sample_data/imagenet_class_labels.txt"
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
def test_deit(device):
    torch.manual_seed(0)

    model_name = "facebook/deit-tiny-patch16-224"
    batch_size = 1
    sequence_size = 224
    iterations = 100

    config = transformers.DeiTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config)
    model.classifier = nn.Linear(192, 10, bias=True)
    model.load_state_dict(
        torch.load("models/demos/wormhole/deit_tiny/demo/deit_tiny_patch16_224_trained_statedict.pth"), strict=True
    )
    config = ttnn_optimized_sharded_deit_wh.update_model_config(config, batch_size)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_deit_wh.custom_preprocessor,
    )

    model.eval()

    model_state_dict = model.state_dict()
    torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
    torch_distillation_token = torch.nn.Parameter(torch.zeros(1, 1, 192))
    torch_position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token.expand(batch_size, -1, -1))
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token)
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    distillation_token = ttnn.from_torch(
        torch_distillation_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )

    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    # CIFAR-10 Inference
    cifar_label_dict = get_cifar10_label_dict()
    data_loader = get_data_loader_cifar10(batch_size=1, iterations=iterations)

    correct_ttnn = 0
    correct_torch = 0
    pccs = []

    for iter in range(iterations):
        predictions_ttnn = []
        predictions_torch = []

        torch_pixel_values, labels = get_batch_cifar(data_loader)

        # Preprocess input for TTNN
        torch_pixel_values_tt = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values_tt = torch.nn.functional.pad(torch_pixel_values_tt, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = torch_pixel_values_tt.shape
        patch_size = 16
        torch_pixel_values_tt = torch_pixel_values_tt.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
        N, H, W, C = torch_pixel_values_tt.shape
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(batch_size - 1, 3),
                ),
            }
        )
        n_cores = batch_size * 3
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

        pixel_values_tt = torch2tt_tensor(
            torch_pixel_values_tt,
            device,
            ttnn.ROW_MAJOR_LAYOUT,
            tt_memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                shard_spec,
            ),
            tt_dtype=ttnn.bfloat16,
        )

        # Run TTNN
        output_ttnn = ttnn_optimized_sharded_deit_wh.deit(
            config,
            pixel_values_tt,
            head_masks,
            cls_token,
            distillation_token,
            position_embeddings,
            parameters=parameters,
        )
        output_ttnn = ttnn.to_torch(output_ttnn)
        prediction_ttnn = output_ttnn[:, 0, :1000].argmax(dim=-1)

        # Run Torch
        with torch.no_grad():
            output_torch = model(torch_pixel_values).logits
            prediction_torch = output_torch.argmax(dim=-1)

        for i in range(batch_size):
            pred_label_ttnn = cifar_label_dict[prediction_ttnn[i].item()]
            pred_label_torch = cifar_label_dict[prediction_torch[i].item()]

            true_label = cifar_label_dict[labels[i]]
            pcc = assert_with_pcc(output_ttnn[:, 0, :1000], output_torch, 0.7)
            pccs.append(pcc[-1])

            if pred_label_ttnn == true_label:
                correct_ttnn += 1
            if pred_label_torch == true_label:
                correct_torch += 1

            if (
                (true_label != pred_label_torch)
                or (true_label != pred_label_ttnn)
                or (pred_label_ttnn != pred_label_torch)
            ):
                logger.warning("some mimatch!")
                logger.warning(
                    f"true_label: {true_label} - pred_label_ttnn: {pred_label_ttnn} - pred_label_torch: {pred_label_torch}"
                )

    accuracy_ttnn = correct_ttnn / (batch_size * iterations)
    accuracy_torch = correct_torch / (batch_size * iterations)
    print(f"CIFAR-10 TTNN Accuracy:  {accuracy_ttnn:.4f}")
    print(f"CIFAR-10 Torch Accuracy: {accuracy_torch:.4f}")
    print("average pcc: ", statistics.mean(pccs))
