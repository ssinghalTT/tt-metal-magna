// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "cpp/ttnn/global_semaphore.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

struct AllReduceCreateQkvHeads {
    std::optional<IDevice*> forward_device;
    std::optional<IDevice*> backward_device;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const MemoryConfig all_reduce_mem_config;
    const ccl::Topology topology;
    const GlobalSemaphore semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    bool enable_persistent_fabric_mode;

    // create qkv heads parameters
    const uint32_t head_dim;
    const uint32_t num_heads;
    const uint32_t num_kv_heads;
    const bool overlap_qk_coregrid;
    const bool input_on_subcoregrids;
    std::optional<const uint32_t> slice_size;
    const MemoryConfig final_mem_config;

    AllReduceCreateQkvHeads(
        std::optional<IDevice*> forward_device,
        std::optional<IDevice*> backward_device,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t ring_index,
        MemoryConfig all_reduce_mem_config,
        ccl::Topology topology,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        bool enable_persistent_fabric_mode,
        uint32_t head_dim,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        bool overlap_qk_coregrid,
        bool input_on_subcoregrids,
        std::optional<const uint32_t> slice_size,
        MemoryConfig final_mem_config) :
        forward_device(forward_device),
        backward_device(backward_device),
        num_links(num_links),
        ring_size(ring_size),
        ring_index(ring_index),
        all_reduce_mem_config(all_reduce_mem_config),
        topology(topology),
        semaphore(semaphore),
        sub_device_id(sub_device_id),
        enable_persistent_fabric_mode(enable_persistent_fabric_mode),
        head_dim(head_dim),
        num_heads(num_heads),
        num_kv_heads(num_kv_heads),
        overlap_qk_coregrid(overlap_qk_coregrid),
        input_on_subcoregrids(input_on_subcoregrids),
        slice_size(slice_size),
        final_mem_config(final_mem_config) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("ring_index", ring_index);
        attrs.emplace_back("all_reduce_mem_config", all_reduce_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);

        // Add the new QKV heads parameters
        attrs.emplace_back("num_heads", num_heads);
        attrs.emplace_back("num_kv_heads", num_kv_heads);
        if (slice_size.has_value()) {
            attrs.emplace_back("slice_size", slice_size.value());
        }
        attrs.emplace_back("final_mem_config", final_mem_config);

        return attrs;
    }

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    const tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

namespace ccl {
namespace all_reduce_create_qkv_heads_detail {
AllReduceCreateQkvHeads create_all_reduce_create_qkv_heads_struct(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool enable_persistent_fabric_mode,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    bool overlap_qk_coregrid,
    bool input_on_subcoregrids,
    std::optional<const uint32_t> slice_size,
    MemoryConfig final_mem_config);

}  // namespace all_reduce_create_qkv_heads_detail
}  // namespace ccl

std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores_fuse(
    size_t num_links,
    size_t num_workers_per_link,
    bool persistent_fabric_mode,
    IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<CoreRangeSet>& reserved_core_range = std::nullopt);

tt::tt_metal::operation::ProgramWithCallbacks all_reduce_create_qkv_heads_minimal_multi_core_with_workers(
    const Tensor& input_tensor,
    const Tensor& buffer_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool enable_persistent_fabric_mode);

namespace operations {
namespace experimental {
namespace ccl {

std::tuple<Tensor, Tensor, Tensor, Tensor> all_reduce_create_qkv_heads(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const Tensor& batch_offset_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& all_reduce_memory_config = std::nullopt,
    const std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    bool enable_persistent_fabric_mode = false,
    uint32_t head_dim = 0,
    uint32_t num_heads = 8,
    uint32_t num_kv_heads = 1,
    bool overlap_qk_coregrid = false,
    bool input_on_subcoregrids = false,
    std::optional<const uint32_t> slice_size = std::nullopt,
    const std::optional<MemoryConfig>& final_memory_config = std::nullopt);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
