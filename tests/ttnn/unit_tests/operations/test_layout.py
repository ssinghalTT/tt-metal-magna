# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


def test_layout_multi_device(mesh_device, reset_seeds):
    weight_tensor = torch.randn(64, 3, 3, 3)  # Conv2d weights tensors shape

    # Replicating the weight_tensor on two devices
    ttnn_input_tensor = ttnn.from_torch(
        weight_tensor, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )

    ttnn_input_tensor = ttnn.from_device(ttnn_input_tensor)
    ttnn_input_tensor = ttnn.to_layout(
        ttnn_input_tensor, ttnn.ROW_MAJOR_LAYOUT
    )  # FAILED # Error: Device storage isn't supported


def test_layout_single_device(device, reset_seeds):
    weight_tensor = torch.randn(64, 3, 3, 3)  # Conv2d weights tensors shape

    ttnn_input_tensor = ttnn.from_torch(weight_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_input_tensor = ttnn.from_device(ttnn_input_tensor)
    ttnn_input_tensor = ttnn.to_layout(ttnn_input_tensor, ttnn.ROW_MAJOR_LAYOUT)


# PASSED
