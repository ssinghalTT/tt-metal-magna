// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/global_cb_utils.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations {

namespace matmul {

using ttnn::operations::unary::UnaryWithParam;

/*
 * GENERAL MATMUL AND BMM
 */
tt::tt_metal::ProgramDescriptor matmul_multi_core(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor& output_tensor, bool bcast_batch);
tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor& output_tensor, bool bcast_batch);
tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_mcast(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor& output_tensor, bool bcast_batch);

tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_mcast_1d_optimized(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    const std::optional<UnaryWithParam>& fused_activation,
    bool mcast_in0,
    bool gather_in0,
    const CoreRangeSet& hop_cores,
    bool untilize_out,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);
tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_dram_sharded_optimized(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back);
tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_mcast_2d_optimized(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out);
tt::tt_metal::ProgramDescriptor bmm_multi_core_reuse_optimized(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    Tensor& output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    tt::tt_metal::DataType output_dtype,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool untilize_out);

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1
// for in1 iff B=1 for in0 (ie. single core)
struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t out_block_h;
    std::size_t out_block_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool transpose_mcast;
    std::optional<UnaryWithParam> fused_activation;
    bool fuse_batch = true;
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t out_block_h;
    std::size_t out_block_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool fuse_batch;
    std::optional<UnaryWithParam> fused_activation;
    bool mcast_in0;
    bool gather_in0;
    CoreRangeSet hop_cores;
    std::size_t num_global_cb_receivers;
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig {
    std::size_t in0_block_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    std::optional<UnaryWithParam> fused_activation;
};

struct MatmulMultiCoreProgramConfig {};

struct MatmulMultiCoreNonOptimizedReuseProgramConfig {};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreNonOptimizedReuseProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>;

struct MatmulArgs {
    const std::optional<const MatmulProgramConfig> program_config = std::nullopt;
    const std::optional<bool> bcast_batch = std::nullopt;
    const MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    const std::optional<DataType> output_dtype = std::nullopt;
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    const bool untilize_out = false;
    const std::optional<const CoreCoord> user_core_coord = std::nullopt;
    const std::optional<UnaryWithParam> user_fused_activation = std::nullopt;
    const bool user_run_batched = false;
    const bool transpose_a = false;
    const bool transpose_b = false;
    const std::optional<const tt::tt_metal::Tile> output_tile;
    const std::optional<const tt::tt_metal::DeviceGlobalCircularBuffer> global_cb;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
};

struct Matmul {
    using operation_attributes_t = MatmulArgs;

    struct tensor_args_t {
        Tensor input_tensor_a;
        Tensor input_tensor_b;
        std::optional<Tensor> bias;
        std::optional<Tensor> output;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static void validate(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::ProgramDescriptor create_program(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModel create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const Tensor>& bias = std::nullopt,
        const MatmulArgs& parameters = {},
        const QueueId queue_id = DefaultQueueId,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

Matmul::operation_attributes_t create_matmul_struct(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Matmul::operation_attributes_t& parameters,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_mcast_1d_optimized_helper(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);
tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_mcast_2d_optimized_helper(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& matmul_signal_info);
}  // namespace matmul

}  // namespace operations

}  // namespace ttnn

namespace bmm_op_utils {

std::tuple<uint32_t, uint32_t> get_matmul_subblock_params(
    const uint32_t per_core_M,
    const uint32_t per_core_N,
    const bool per_core_M_equals_subblock_h_constraint,
    const bool per_core_N_equals_subblock_w_constraint,
    const bool fp32_dest_acc_en);

void add_stagger_defines_if_needed(
    const tt::ARCH arch, const int num_cores, tt::tt_metal::KernelDescriptor::Defines& mm_kernel_defines);

}  // namespace bmm_op_utils

namespace ttnn::prim {
constexpr auto matmul = ttnn::register_operation<"ttnn::prim::matmul", ttnn::operations::matmul::Matmul>();
}  // namespace ttnn::prim
