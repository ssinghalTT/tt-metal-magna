// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/dispatch/memcpy.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "tt_metal/impl/sub_device/sub_device.hpp"
#include "tt_metal/tt_stl/span.hpp"

namespace tt::tt_metal {

inline namespace v0 {
class Device;
}  // namespace v0
namespace detail {
class SubDeviceManager {
   public:
    SubDeviceManager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size, Device *device);
    SubDeviceManager(std::vector<SubDevice>&& sub_devices, DeviceAddr local_l1_size, Device *device);

    SubDeviceManager(const SubDeviceManager &other) = delete;
    SubDeviceManager &operator=(const SubDeviceManager &other) = delete;

    SubDeviceManager(SubDeviceManager &&other) = default;
    SubDeviceManager &operator=(SubDeviceManager &&other) = default;

    ~SubDeviceManager();

    const SubDevice &sub_device(uint32_t sub_device_id) const;
    const vector_memcpy_aligned<uint32_t> &noc_mcast_data(uint32_t sub_device_id) const;
    const vector_memcpy_aligned<uint32_t> &noc_unicast_data(uint32_t sub_device_id) const;
    const vector_memcpy_aligned<uint32_t> &noc_mcast_unicast_data(uint32_t sub_device_id) const;
    std::unique_ptr<Allocator> &sub_device_allocator(uint32_t sub_device_id);
    const std::unordered_set<uint32_t> &trace_ids() const;
    void add_trace_id(uint32_t trace_id);
    void remove_trace_id(uint32_t trace_id);

    uint32_t num_sub_devices() const;
    bool has_allocations() const;
    DeviceAddr local_l1_size() const;

    // friend class tt::tt_metal::Device;

   private:
    void validate_sub_devices() const;
    void populate_num_cores();
    void populate_sub_allocators();
    void populate_noc_data();

    std::vector<SubDevice> sub_devices_;
    DeviceAddr local_l1_size_;
    Device *device_;
    std::vector<std::unique_ptr<Allocator>> sub_device_allocators_;
    std::unordered_set<uint32_t> trace_ids_;
    std::array<uint32_t, NumHalProgrammableCoreTypes> num_cores_{};
    std::vector<vector_memcpy_aligned<uint32_t>> noc_mcast_data_;
    std::vector<vector_memcpy_aligned<uint32_t>> noc_unicast_data_;
    // Concatenation of noc_mcast_data_ and noc_unicast_data_
    // Useful for optimized copying of all coords when constructing FD commands
    std::vector<vector_memcpy_aligned<uint32_t>> noc_mcast_unicast_data_;
};

}  // namespace detail

using SubDeviceManagerId = uint64_t;

}  // namespace tt_metal
