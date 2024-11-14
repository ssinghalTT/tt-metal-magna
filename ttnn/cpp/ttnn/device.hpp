// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"
namespace ttnn {

namespace device {

using Device = ttnn::Device;

Device &open_device(int device_id, size_t l1_small_size = DEFAULT_L1_SMALL_SIZE, size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE, tt::tt_metal::DispatchCoreType dispatch_core_type = tt::tt_metal::DispatchCoreType::WORKER);
void close_device(Device &device);
void enable_program_cache(Device &device);
void disable_and_clear_program_cache(Device &device);
bool is_wormhole_or_blackhole(tt::ARCH arch);
void deallocate_buffers(Device *device);
// TODO: Change to taking in tt::stl::Span once there is a pybind for it
SubDeviceManagerId create_sub_device_manager(Device *device, const std::vector<SubDevice> &sub_devices, DeviceAddr local_l1_size);
void load_sub_device_manager(Device *device, SubDeviceManagerId sub_device_manager_id);
void reset_active_sub_device_manager(Device *device);
void remove_sub_device_manager(Device *device, SubDeviceManagerId sub_device_manager_id);

}  // namespace device

using namespace device;

}  // namespace ttnn
