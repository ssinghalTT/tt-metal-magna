// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_create_qkv_heads_op.hpp"
#include "ttnn/operations/math.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

#include <tt-metalium/host_api.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
namespace ccl {
namespace all_reduce_create_qkv_heads_detail {

AllReduceCreateQkvHeads create_all_reduce_create_qkv_heads_struct(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& all_reduce_memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool enable_persistent_fabric_mode,
    uint32_t head_dim,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    bool overlap_qk_coregrid,
    bool input_on_subcoregrids,
    std::optional<const uint32_t> slice_size,
    const std::optional<MemoryConfig>& final_memory_config) {
    uint32_t num_devices = devices.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    std::optional<GlobalSemaphore> semaphore = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            device_index = i;
            semaphore = semaphores.at(i);  // Get raw pointer
            if (i != 0) {
                backward_device = devices.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices.at(num_devices - 1);
            }
            if (i != num_devices - 1) {
                forward_device = devices.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices.at(0);
            }
        }
    }

    return ttnn::AllReduceCreateQkvHeads{
        forward_device,
        backward_device,
        num_links,
        num_devices,
        device_index,
        all_reduce_memory_config.value_or(input_tensor.memory_config()),
        topology,
        semaphore.value(),
        sub_device_id,
        enable_persistent_fabric_mode,
        head_dim,
        num_heads,
        num_kv_heads,
        overlap_qk_coregrid,
        input_on_subcoregrids,
        slice_size,
        final_memory_config.value_or(input_tensor.memory_config())};
}

}  // namespace all_reduce_create_qkv_heads_detail
}  // namespace ccl

void AllReduceCreateQkvHeads::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Error, Input tensor size should be 3 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& buffer_tensor = input_tensors[1];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");
    TT_FATAL(
        this->ring_size % 2 == 0,
        "AllReduceAsync currently only supports even number of blocks in the reduction kernel.");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");

    TT_FATAL(buffer_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(buffer_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");

    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for input tensor{}.",
        input_tensor.memory_config().memory_layout);

    TT_FATAL(
        buffer_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for buffer tensor {}.",
        buffer_tensor.memory_config().memory_layout);
    TT_FATAL(
        this->all_reduce_mem_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for output tensor {}.",
        this->all_reduce_mem_config.memory_layout);

    TT_FATAL(
        buffer_tensor.memory_config().shard_spec->grid.contains(this->all_reduce_mem_config.shard_spec->grid),
        "The output tensor must reside on a subset of the cores of the buffer tensor");

    const uint32_t output_shard_shape_volume =
        this->all_reduce_mem_config.shard_spec->shape[0] * this->all_reduce_mem_config.shard_spec->shape[1];
    const uint32_t buffer_shard_shape_volume =
        buffer_tensor.memory_config().shard_spec->shape[0] * buffer_tensor.memory_config().shard_spec->shape[1];
    TT_FATAL(
        output_shard_shape_volume * this->ring_size <= buffer_shard_shape_volume,
        "The shard size for the buffer must be large enough to hold the intermediate tensor. Require at least {} but "
        "has {}",
        output_shard_shape_volume * this->ring_size,
        buffer_shard_shape_volume);

    // validate for create qkv heads
    const auto& input_shape = input_tensor.get_logical_shape();
    const auto& batch_offset = input_tensors.at(2);

    // TODO: Rewrite validation for this decode case
    // NOTE: Checks for head_dim and shape[3] is done in nlp_create_qkv_heads because it's needed to infer head_dim
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16,
        "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Only tile layout is supported for input tensor");

    // input
    const uint32_t num_users_supported = 32;
    uint32_t num_users = input_shape[2];
    TT_FATAL(
        input_shape[3] % TILE_WIDTH == 0,
        "Unsupported input shape = {}",
        input_shape);  // head_dim must be multiple of TILE_WIDTH
    TT_FATAL(num_users <= num_users_supported, "Unsupported input shape = {}", input_shape);  // 32 users
    TT_FATAL(input_shape[1] == 1, "Unsupported input shape = {}", input_shape);
    TT_FATAL(input_shape[0] == 1, "Unsupported input shape = {}", input_shape);
    const auto QKV_memcfg = input_tensor.memory_config();
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            QKV_memcfg.memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
            "Current input memory layout is {}. It must be width sharded",
            QKV_memcfg.memory_layout);
        TT_FATAL(
            input_tensor.shard_spec().value().shape[0] == input_tensor.volume() / input_tensor.get_padded_shape()[-1],
            "Shard shape must be correct");
        TT_FATAL(
            input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
            "Shard orientation must be ROW_MAJOR");

        if (!this->overlap_qk_coregrid) {
            // Validate if each shard is a multiple of head_dim and doesn't contain partial heads
            TT_FATAL(
                this->head_dim % input_tensor.shard_spec().value().shape[1] == 0,
                "We don't support partial heads in shards when q and k heads are not overlapping coregrid");
        }
        /* Don't validate batch_offset and slice_size for now, as they will be provided by the user
        TT_FATAL(
            !(batch_offset.has_value() ^ this->slice_size.has_value()),
            "Both batch_offset and slice_size must be provided or neither");
        if (batch_offset.has_value() && this->slice_size.has_value()) {
            TT_FATAL(batch_offset.value().get_logical_shape()[0] == 1, "batch_offset must be unary tensor");
            num_users = this->slice_size.value();
        }
        */

    } else {
        TT_FATAL(this->overlap_qk_coregrid, "Overlap_qk_coregrid must be true for non-sharded input");
    }

    // output
    TT_FATAL(
        this->final_mem_config.is_sharded() &&
            this->final_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Output tensor must be height sharded");

    // Support maximum 32 heads for now
    TT_FATAL(this->num_heads <= 32, "There are {} q heads only 32 are supported", this->num_heads);
    TT_FATAL(
        this->num_heads >= this->num_kv_heads,
        "num_q_heads={} must be greater than or equal to num_kv_heads={}",
        this->num_heads,
        this->num_kv_heads);

    uint32_t num_cores;
    if (this->input_on_subcoregrids) {
        auto input_core_grid = input_tensor.shard_spec().value().grid;
        num_cores = input_core_grid.num_cores();

    } else {
        auto core_grid_size = input_tensor.device()->compute_with_storage_grid_size();
        num_cores = core_grid_size.x * core_grid_size.y;
    }
    // 1 User Per Core Max and 32 users for now
    if (this->overlap_qk_coregrid) {
        TT_FATAL(num_cores >= num_users, "Grid Size is {}. Need at least 32 cores for decode", num_cores);
    } else {
        TT_FATAL(
            num_cores >= 2 * num_users,
            "Input coregrid size is {}. Need cores atleast double of num_users for decode when q and k heads are not "
            "overlapping "
            "coregrid",
            num_cores);
    }
}

std::vector<ttnn::TensorSpec> AllReduceCreateQkvHeads::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    const auto& input_shape = input_tensor.get_logical_shape();
    auto output_tensor_layout =
        input_tensor.get_tensor_spec().tensor_layout().with_memory_config(this->all_reduce_mem_config);
    auto all_reduce_tensor_spec{TensorSpec(input_shape, output_tensor_layout)};
    // return {all_reduce_tensor_spec, all_reduce_tensor_spec, all_reduce_tensor_spec, all_reduce_tensor_spec};
    // copied from qkv create heads compute_output_specs
    using namespace tt::constants;
    // const auto& input_tensor = input_tensors.at(0);
    // const auto& input_shape = input_tensor.get_logical_shape();

    auto batch = input_shape[2];
    if (this->slice_size.has_value()) {
        batch = this->slice_size.value();
    }

    auto head_dim = this->head_dim;

    const Shape q_output_shape({input_shape[0], batch, this->num_heads, head_dim});
    const Shape v_output_shape({input_shape[0], batch, this->num_kv_heads, head_dim});
    const Shape k_output_shape = v_output_shape;

    auto num_q_heads_padded = ((this->num_heads - 1) / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto num_kv_heads_padded = ((this->num_heads - 1) / TILE_HEIGHT + 1) * TILE_HEIGHT;

    MemoryConfig q_mem_config = this->final_mem_config;
    MemoryConfig k_mem_config = this->final_mem_config;
    MemoryConfig v_mem_config = this->final_mem_config;
    CoreRangeSet q_shard_grid, k_shard_grid, v_shard_grid;
    // auto input_core_grid = input_tensor.shard_spec().value().grid;
    auto sub_core_grid = tt::tt_metal::CoreRangeSet(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(1, 0), tt::tt_metal::CoreCoord(3, 9)));
    auto start_core_coord = sub_core_grid.bounding_box().start_coord;
    auto next_core_coord = start_core_coord;

    q_shard_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch, sub_core_grid, true);

    CoreRangeSet q_batch_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch + 1, sub_core_grid, true);
    if (!q_batch_grid.ranges().empty()) {
        next_core_coord = q_batch_grid.ranges().back().end_coord;
    }
    k_shard_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(next_core_coord, batch, sub_core_grid, true);

    CoreRangeSet q_two_batch_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, 2 * batch + 1, sub_core_grid, true);
    if (!q_two_batch_grid.ranges().empty()) {
        next_core_coord = q_two_batch_grid.ranges().back().end_coord;
    }
    v_shard_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(next_core_coord, batch, sub_core_grid, true);

    tt::tt_metal::ShardSpec q_shard_spec{q_shard_grid, {num_q_heads_padded, this->head_dim}};
    q_mem_config.shard_spec = q_shard_spec;
    tt::tt_metal::ShardSpec k_shard_spec{k_shard_grid, {num_kv_heads_padded, this->head_dim}};
    k_mem_config.shard_spec = k_shard_spec;
    tt::tt_metal::ShardSpec v_shard_spec{v_shard_grid, {num_kv_heads_padded, this->head_dim}};
    v_mem_config.shard_spec = v_shard_spec;

    return {
        all_reduce_tensor_spec,
        TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), q_mem_config)),
        TensorSpec(
            k_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), k_mem_config)),
        TensorSpec(
            v_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), v_mem_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks AllReduceCreateQkvHeads::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");

    auto input_tensor_shape = input_tensors[0].get_padded_shape();
    auto input_tensor_buffer_layout = input_tensors[0].buffer()->buffer_layout();
    auto input_tensor_page_layout = input_tensors[0].layout();

    auto input_tensor_memory_config = input_tensors[0].memory_config();
    auto output_tensor_memory_config = output_tensors[0].memory_config();
    uint32_t input_shard_num_cores = input_tensor_memory_config.shard_spec->grid.num_cores();
    uint32_t output_shard_num_cores = output_tensor_memory_config.shard_spec->grid.num_cores();

    tt::log_debug(tt::LogOp, "input_tensor_shape: {}", input_tensor_shape);
    tt::log_debug(tt::LogOp, "input_tensor_memory_config: {}", input_tensor_memory_config);
    tt::log_debug(tt::LogOp, "output_tensor_memory_config: {}", output_tensor_memory_config);
    tt::log_debug(tt::LogOp, "input_shard_num_cores: {}", input_shard_num_cores);
    tt::log_debug(tt::LogOp, "output_shard_num_cores: {}", output_shard_num_cores);
    tt::log_debug(
        tt::LogOp, "input_tensor_memory_config.shard_spec->shape: {}", input_tensor_memory_config.shard_spec->shape);
    tt::log_debug(
        tt::LogOp, "output_tensor_memory_config.shard_spec->shape: {}", output_tensor_memory_config.shard_spec->shape);

    tt::log_debug(tt::LogOp, "Running TG Llama specific all_reduce_create_qkv_heads_minimal_multi_core_with_workers");
    return all_reduce_create_qkv_heads_minimal_multi_core_with_workers(
        input_tensors,
        this->forward_device,
        this->backward_device,
        output_tensors,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->topology,
        this->semaphore,
        this->sub_device_id,
        this->enable_persistent_fabric_mode,
        this->num_heads,
        this->num_kv_heads,
        this->head_dim);
}

const tt::tt_metal::operation::Hash AllReduceCreateQkvHeads::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<AllReduceCreateQkvHeads>(
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->all_reduce_mem_config,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

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
    const std::optional<MemoryConfig>& all_reduce_memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode,
    uint32_t head_dim,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    bool overlap_qk_coregrid,
    bool input_on_subcoregrids,
    std::optional<const uint32_t> slice_size,
    const std::optional<MemoryConfig>& final_memory_config) {
    const auto mesh_view = mesh_device.get_view();
    auto devices = input_tensor.get_workers();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    std::vector<Tensor> output_tensors{
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor})),
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor})),
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor})),
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;

    tt::tt_metal::operation::launch_op(
        [num_preferred_links,
         all_reduce_memory_config,
         mesh_view,
         cluster_axis,
         num_devices,
         topology,
         semaphores,
         subdevice_id,
         enable_persistent_fabric_mode,
         head_dim,
         num_heads,
         num_kv_heads,
         overlap_qk_coregrid,
         input_on_subcoregrids,
         slice_size,
         final_memory_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_device_tensor = input_tensors.at(0);

            TT_FATAL(
                mesh_view.is_mesh_2d(),
                "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
            const auto coordinate = mesh_view.find_device(input_device_tensor.device()->id());
            std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(coordinate[1])
                                                                : mesh_view.get_devices_on_row(coordinate[0]);

            const auto& input_tensor = input_tensors.at(0);
            const auto& buffer_tensor = input_tensors.at(1);
            const auto& batch_offset_tensor = input_tensors.at(2);

            return tt::tt_metal::operation::run(
                ttnn::ccl::all_reduce_create_qkv_heads_detail::create_all_reduce_create_qkv_heads_struct(
                    input_device_tensor,
                    num_preferred_links.has_value() ? num_preferred_links.value() : 1,
                    all_reduce_memory_config,
                    devices,
                    topology,
                    semaphores,
                    subdevice_id,
                    enable_persistent_fabric_mode,
                    head_dim,
                    num_heads,
                    num_kv_heads,
                    overlap_qk_coregrid,
                    input_on_subcoregrids,
                    slice_size,
                    final_memory_config),
                {input_tensor, buffer_tensor, batch_offset_tensor});
        },
        {input_tensor, buffer_tensor, batch_offset_tensor},
        output_tensors);
    return std::make_tuple(output_tensors.at(0), output_tensors.at(1), output_tensors.at(2), output_tensors.at(3));
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores_fuse(
    size_t num_links,
    size_t num_workers_per_link,
    bool persistent_fabric_mode,
    IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<CoreRangeSet>& reserved_core_range) {
    std::tuple<CoreRangeSet, std::vector<CoreCoord>> result;
    CoreRangeSet sender_worker_core_range;
    if (persistent_fabric_mode) {
        const size_t num_workers_preferred = num_workers_per_link * num_links;
        auto available_cores = device->worker_cores(
            tt::tt_metal::HalProgrammableCoreType::TENSIX,
            sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
        if (reserved_core_range.has_value()) {
            available_cores = available_cores.subtract(*reserved_core_range);
        }
        if (available_cores.num_cores() < num_workers_preferred) {
            log_warning(
                tt::LogOp,
                "AllGather is being launched on a subdevice with fewer worker cores available than ideal. Ideally {} "
                "cores ({} per link and {} links) are made available but only {} are available. This may lead to "
                "performance loss.",
                num_workers_preferred,
                num_workers_per_link,
                num_links,
                available_cores.num_cores());
        }
        for (const auto& cr : available_cores.ranges()) {
            auto start = cr.start_coord;
            auto end = cr.end_coord;
            for (size_t y = start.y; y <= end.y; y++) {
                for (size_t x = start.x; x <= end.x; x++) {
                    sender_worker_core_range =
                        sender_worker_core_range.merge(CoreRangeSet(CoreRange(CoreCoord(x, y), CoreCoord(x, y))));
                    if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                        break;
                    }
                }
                if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                    break;
                }
            }
            if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                break;
            }
        }
    } else {
        sender_worker_core_range =
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_workers_per_link - 1, num_links - 1)));
    }
    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
}

}  // namespace ttnn
