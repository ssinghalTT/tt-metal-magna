# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import json
import numpy as np
import os
from loguru import logger
from models.experimental.ufld_v2_rn18like.reference.ufld_v2_rn18like_model import TuSimple18like
from models.experimental.ufld_v2_rn18like.demo import model_config as cfg
from models.experimental.ufld_v2_rn18like.demo.demo_utils import (
    run_test_tusimple,
    LaneEval,
)
from models.experimental.ufld_v2_rn18like.tt.ttnn_ufld_v2_rn18like import (
    TtnnUFLDV2RN18like,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    infer_ttnn_module_args,
)
from models.experimental.ufld_v2_rn18like.tests.test_ttnn_ufld_v2_rn18like import custom_preprocessor


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 320, 800),
        # (2, 3, 320, 800),
        # (4, 3, 320, 800),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        # False,
        True
    ],
    ids=[
        # "pretrained_weight_false",
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ufld_v2_res18like_inference(batch_size, input_channels, height, width, device, use_pretrained_weight):
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width))
    reference_model = TuSimple18like(input_height=height, input_width=width)
    if use_pretrained_weight:
        logger.info(f"Demo Inference using Pre-trained Weights")
        weights_path = "models/experimental/ufld_v2_rn18like/demo/ufldv2_resnet18_trained_statedict.pth"
        state_dict = torch.load(weights_path)
        reference_model.load_state_dict(state_dict)
    else:
        logger.info(f"Demo Inference using Random Weights")
    cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    run_test_tusimple(
        reference_model,
        cfg.data_root,
        cfg.data_root,
        "reference_model_results",
        False,
        cfg.crop_ratio,
        cfg.train_width,
        cfg.train_height,
        batch_size=batch_size,
        row_anchor=cfg.row_anchor,
        col_anchor=cfg.col_anchor,
        device=None,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=reference_model, run_model=lambda model: reference_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnUFLDV2RN18like(conv_args=parameters.conv_args, conv_pth=parameters, device=device)
    run_test_tusimple(
        ttnn_model,
        cfg.data_root,
        cfg.data_root,
        "ttnn_model_results",
        False,
        cfg.crop_ratio,
        cfg.train_width,
        cfg.train_height,
        batch_size=batch_size,
        row_anchor=cfg.row_anchor,
        col_anchor=cfg.col_anchor,
        device=device,
    )

    gt_file_path = os.path.join(cfg.data_root, "GT_test_labels" + ".json")
    res = LaneEval.bench_one_submit(os.path.join(cfg.data_root, "reference_model_results" + ".txt"), gt_file_path)
    res = json.loads(res)
    for r in res:
        if r["name"] == "F1":
            logger.info(f"F1 Score for Reference Model is {r['value']}")

    res1 = LaneEval.bench_one_submit(os.path.join(cfg.data_root, "ttnn_model_results" + ".txt"), gt_file_path)
    res1 = json.loads(res1)
    for r in res1:
        if r["name"] == "F1":
            logger.info(f"F1 Score for ttnn Model is {r['value']}")
