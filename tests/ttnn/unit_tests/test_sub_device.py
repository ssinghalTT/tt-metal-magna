# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def test_subdevice(device):
    cores = device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord((cores.x - 1) // 2, (cores.y - 1) // 2))}
    )
    sub_device = ttnn.SubDevice([crs])
    sub_device_manager = ttnn.create_sub_device_manager(device, [sub_device], 160)
    ttnn.load_sub_device_manager(device, sub_device_manager)
    ttnn.reset_active_sub_device_manager(device)
    ttnn.remove_sub_device_manager(device, sub_device_manager)
