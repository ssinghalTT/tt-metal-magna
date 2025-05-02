# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.centernet.reference.resnet import BasicBlock, ResNet
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_torch_backbone():
    with torch.no_grad():
        reference = ResNet(BasicBlock, [2, 2, 2, 2])
        reference.eval()
    print(reference)
    state_dict = torch.load("models/experimental/centernet/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth")[
        "state_dict"
    ]
    new_state_dict = {}
    for k, v in state_dict.items():
        if "backbone" in k:
            new_state_dict[k[9:]] = v
    reference.load_state_dict(new_state_dict)
    # input = torch.rand(1, 3, 32, 32)
    input = torch.load("backbone_centernet_input_model.pt")
    ouput = reference.forward(input)
    assert_with_pcc(torch.load("one.pt"), torch.load("torch_one.pt"), 0.99)
    assert_with_pcc(torch.load("two.pt"), torch.load("torch_two.pt"), 0.99)
    assert_with_pcc(torch.load("three.pt"), torch.load("torch_three.pt"), 0.99)
    assert_with_pcc(torch.load("Enter.pt"), torch.load("torch_enter.pt"), 0.99)

    assert_with_pcc(torch.load("torch_backbone_out_0.pt"), torch.load("backbone_out_0.pt"), 0.99)
    assert_with_pcc(torch.load("torch_backbone_out_1.pt"), torch.load("backbone_out_1.pt"), 0.99)
    assert_with_pcc(torch.load("torch_backbone_out_2.pt"), torch.load("backbone_out_2.pt"), 0.99)
    assert_with_pcc(torch.load("torch_backbone_out_3.pt"), torch.load("backbone_out_3.pt"), 0.99)

    assert_with_pcc(ouput[0], torch.load("backbone_out_0.pt"), 0.99)
    assert_with_pcc(ouput[1], torch.load("backbone_out_1.pt"), 0.99)
    assert_with_pcc(ouput[2], torch.load("backbone_out_2.pt"), 0.99)
    assert_with_pcc(ouput[3], torch.load("backbone_out_3.pt"), 0.99)
