// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt;
using namespace tt::constants;
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace reuse_mcast_1d_optimized_helpers {

uint32_t get_preferred_noc(
    const ttnn::CoreCoord src,
    const ttnn::CoreCoord dst,
    const tt_metal::IDevice* device,
    const bool use_dedicated_noc = false) {
    /*
        NOC0: Preferred +x -> +y
        NOC1: Preferred -y -> -x
    */

    uint32_t src_x = src.x, src_y = src.y;
    uint32_t dst_x = dst.x, dst_y = dst.y;

    uint32_t MAX_X = device->grid_size().x;
    uint32_t MAX_Y = device->grid_size().y;

    // Get the wrapped distances
    uint32_t dist_right = src_x <= dst_x ? dst_x - src_x : MAX_X - src_x + dst_x;
    uint32_t dist_left = src_x < dst_x ? src_x + MAX_X - dst_x : src_x - dst_x;

    uint32_t dist_bottom = src_y <= dst_y ? dst_y - src_y : MAX_Y - src_y + dst_y;
    uint32_t dist_top = src_y < dst_y ? src_y + MAX_Y - dst_y : src_y - dst_y;

    uint32_t dist_noc_0 = dist_right + dist_bottom;
    uint32_t dist_noc_1 = dist_top + dist_left;

    uint32_t noc = dist_noc_0 < dist_noc_1 ? 0 : 1;

    // Debug print if needed
    // std::cout << "src: (" << src_x << ", " << src_y << "), dst: (" << dst_x << ", " << dst_y << "), noc: " << noc <<
    // std::endl;

    return use_dedicated_noc ? 1 : noc;
}

tt::tt_metal::ProgramDescriptor create_program_mcast_in0(
    const tt::tt_metal::Tensor& a,
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    CoreCoord compute_with_storage_grid_size,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* bias_buffer,
    tt_metal::Buffer* out_buffer,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool in0_is_sharded,
    bool in1_is_sharded,
    bool bias_is_sharded,
    bool output_is_sharded,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
    using tt::tt_metal::num_cores_to_corerangeset;

    tt_metal::ProgramDescriptor program;

    // currently only support transpose of the full tile
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    bool fuse_op = fused_op_signaler.has_value();

    uint32_t num_blocks = K / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2;  // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    uint32_t in0_shard_height_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
        in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0];
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }
    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 2;  // double buffer
    }
    if (in1_is_sharded) {
        uint32_t in1_shard_height_in_tiles = in1_buffer->shard_spec().shape()[0] / in1_tile.get_tile_shape()[0];
        in1_CB_tiles = per_core_N * in1_shard_height_in_tiles;
    }

    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_shard_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    if (output_is_sharded) {
        out_CB_tiles = out_shard_tiles;
    }
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    CoreCoord start_core = {0, 0};
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = compute_with_storage_grid_size.x;

    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores_with_work = num_blocks_total;

    uint32_t in0_sender_num_cores = in0_is_sharded ? a.shard_spec().value().grid.num_cores() : 1;
    uint32_t num_cores = in0_is_sharded ? std::max(num_cores_with_work, in0_sender_num_cores) : num_cores_with_work;

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset(start_core, num_cores, compute_with_storage_grid_size, row_major);

    CoreRangeSet in0_mcast_sender_cores =
        num_cores_to_corerangeset(in0_sender_num_cores, compute_with_storage_grid_size, row_major);
    CoreCoord in0_mcast_sender_cores_grid = in0_mcast_sender_cores.bounding_box().grid_size();

    CoreRangeSet all_cores_with_work =
        num_cores_to_corerangeset(num_cores_with_work, compute_with_storage_grid_size, row_major);
    CoreRange in0_mcast_receiver_cores_bounding_box = all_cores_with_work.bounding_box();
    uint32_t in0_mcast_receiver_num_cores = in0_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid
    uint32_t in0_mcast_receiver_num_dests = std::min(
        in0_mcast_receiver_num_cores,
        num_cores);  // should always be number of cores in receiver grid up to number of active cores

    CoreRangeSet in0_mcast_cores_with_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_not_in_receiver_grid;
    CoreRangeSet in0_mcast_receivers;
    boost::container::small_vector<uint32_t, 16> in0_mcast_noc_x;
    boost::container::small_vector<uint32_t, 16> in0_mcast_noc_y;
    if (in0_is_sharded) {
        in0_mcast_cores_with_work_and_in_receiver_grid = all_cores_with_work;

        if (in0_mcast_receiver_num_dests > num_cores_with_work) {
            const uint32_t in0_mcast_cores_without_work_and_in_receiver_grid_num_cores =
                in0_mcast_receiver_num_dests - num_cores_with_work;
            uint32_t core_idx_x = num_cores_with_work % num_cores_c;
            uint32_t core_idx_y = num_cores_with_work / num_cores_c;
            CoreCoord start_core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            in0_mcast_cores_without_work_and_in_receiver_grid = num_cores_to_corerangeset(
                start_core,
                in0_mcast_cores_without_work_and_in_receiver_grid_num_cores,
                compute_with_storage_grid_size,
                row_major);
        }

        if (in0_sender_num_cores > in0_mcast_receiver_num_dests) {
            const uint32_t in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores =
                in0_sender_num_cores - in0_mcast_receiver_num_dests;
            uint32_t core_idx_x = in0_mcast_receiver_num_dests % num_cores_c;
            uint32_t core_idx_y = in0_mcast_receiver_num_dests / num_cores_c;
            CoreCoord start_core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            in0_mcast_cores_without_work_and_not_in_receiver_grid = num_cores_to_corerangeset(
                start_core,
                in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores,
                compute_with_storage_grid_size,
                row_major);
        }

        in0_mcast_noc_x.reserve(in0_mcast_sender_cores_grid.x);
        in0_mcast_noc_y.reserve(in0_mcast_sender_cores_grid.y);
        for (uint32_t core_idx_x = 0; core_idx_x < in0_mcast_sender_cores_grid.x; ++core_idx_x) {
            in0_mcast_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
        }
        for (uint32_t core_idx_y = 0; core_idx_y < in0_mcast_sender_cores_grid.y; ++core_idx_y) {
            in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
        }
    } else {
        in0_mcast_cores_with_work_and_in_receiver_grid = CoreRangeSet({CoreRange(start_core, start_core)});
        if (in0_mcast_receiver_num_cores > 1) {
            auto receiver_start_core = start_core.x != (compute_with_storage_grid_size.x - 1)
                                           ? CoreCoord{start_core.x + 1, start_core.y}
                                           : CoreCoord{start_core.x, start_core.y + 1};
            in0_mcast_receivers = num_cores_to_corerangeset(
                receiver_start_core, num_cores - 1, compute_with_storage_grid_size, row_major);
        }
    }

    // Mcast args
    uint32_t in0_mcast_sender_semaphore_id = program.add_semaphore(all_cores.ranges(), INVALID);
    uint32_t in0_mcast_receiver_semaphore_id = program.add_semaphore(all_cores.ranges(), INVALID);

    CoreCoord top_left_core = in0_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in0_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    bool in3_is_dram = true;
    if (bias_buffer != nullptr) {
        in3_is_dram = bias_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    }
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    tt_metal::KernelDescriptor::CompileTimeArgs in0_sender_compile_time_args;
    if (in0_is_sharded) {
        in0_sender_compile_time_args = {
            (std::uint32_t)1,  // core_has_output_block_work
            (std::uint32_t)1,  // core_in_in0_receiver_mcast_grid

            (std::uint32_t)in0_block_num_tiles,                         // in0_block_num_tiles
            (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,  // in0_block_size_bytes
            // in0/in1 common args
            (std::uint32_t)num_blocks,        // num_blocks
            (std::uint32_t)out_num_blocks_x,  // num_blocks_x
            (std::uint32_t)out_num_blocks_y,  // num_blocks_y
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_num_dests,  // in0_mcast_num_dests
            (std::uint32_t)in0_mcast_receiver_num_cores,  // in0_mcast_num_cores
            (std::uint32_t)(in0_mcast_sender_cores_grid.x),
            (std::uint32_t)(in0_mcast_sender_cores_grid.y),
            (std::uint32_t)(false),
            (std::uint32_t)(in0_shard_width_in_tiles),
            (std::uint32_t)(in0_shard_height_in_tiles),
            (std::uint32_t)(in0_block_w),
            (std::uint32_t)in0_block_h,  // in0_block_h

            // batch args
            (std::uint32_t)B  // batch
        };
    } else {
        in0_sender_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t)in0_is_dram,

            // in0 tensor args
            (std::uint32_t)1,                // in0_tensor_stride_w
            (std::uint32_t)K,                // in0_tensor_stride_h
            (std::uint32_t)in0_block_w,      // in0_tensor_next_block_stride
            (std::uint32_t)K * in0_block_h,  // in0_tensor_next_h_dim_block_stride
            // in0 block args
            (std::uint32_t)in0_block_w,          // in0_block_w
            (std::uint32_t)in0_block_h,          // in0_block_h
            (std::uint32_t)in0_block_num_tiles,  // in0_block_num_tiles
            (std::uint32_t)false,                // extract_shard_sub_blocks (not used for interleaved)
            (std::uint32_t)0,                    // shard_width_in_tiles (not used for interleaved)
            (std::uint32_t)0,                    // shard_height_in_tiles (not used for interleaved)
            // in0/in1 common args
            (std::uint32_t)num_blocks,        // num_blocks
            (std::uint32_t)out_num_blocks_x,  // num_blocks_x
            (std::uint32_t)out_num_blocks_y,  // num_blocks_y
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)num_cores - 1,                     // in0_mcast_num_dests
            (std::uint32_t)in0_mcast_receiver_num_cores - 1,  // in0_mcast_num_cores
            // batch args
            (std::uint32_t)M * K,  // MtKt
            (std::uint32_t)B       // batch
        };
    }
    in0_sender_compile_time_args.push_back((std::uint32_t)fuse_op);

    tt_metal::KernelDescriptor::Defines mm_kernel_defines;
    tt_metal::KernelDescriptor::Defines mm_kernel_in0_sender_writer_defines;
    tt_metal::KernelDescriptor::Defines mm_kernel_in1_sender_writer_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines.emplace_back("FUSE_BIAS", "1");
        mm_kernel_in1_sender_writer_defines.emplace_back("FUSE_BIAS", "1");
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines.emplace_back("PACK_RELU", "1");
        } else {
            using ttnn::operations::unary::utils::get_defines_vec;
            auto extra_defines =
                get_defines_vec(fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i");
            mm_kernel_defines.insert(mm_kernel_defines.end(), extra_defines.begin(), extra_defines.end());
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines.emplace_back("PACKER_L1_ACC", "1");
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    if (in1_transpose_tile) {
        mm_kernel_defines.emplace_back("IN1_TRANSPOSE_TILE", "1");
    }

    bmm_op_utils::add_stagger_defines_if_needed(device->arch(), num_cores, mm_kernel_defines);

    if (in1_is_sharded) {
        mm_kernel_in1_sender_writer_defines.emplace_back("IN1_SHARDED", "1");
    }

    if (bias_is_sharded) {
        mm_kernel_in1_sender_writer_defines.emplace_back("BIAS_SHARDED", "1");
    }

    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines.emplace_back("OUT_SHARDED", "1");
    }

    // TODO: SKIP_MCAST flag isn't used for the sharded reader kernel because internal mcast logic already works without
    // skipping We can use this flag to turn off unnecessary mcast overhead if necessary
    if (in0_mcast_receiver_num_cores == 1) {
        mm_kernel_in0_sender_writer_defines.emplace_back("SKIP_MCAST", "1");
    }

    mm_kernel_in1_sender_writer_defines.emplace_back("SKIP_MCAST", "1");

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());

    if (fuse_op) {
        // Create semaphores
        fused_op_signaler->init_fused_op(
            program,
            device,
            in0_mcast_sender_cores,
            in0_is_sharded ? ttnn::experimental::ccl::FusedOpSignalerMode::SINGLE
                           : ttnn::experimental::ccl::FusedOpSignalerMode::MULTI);
    }

    constexpr auto max_num_kernels = 6;
    program.kernels.resize(max_num_kernels);
    size_t num_kernels = 0;

    auto& mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid = program.kernels[num_kernels++];
    mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid.kernel_source =
        in0_is_sharded ? "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                         "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp"
                       : "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                         "reader_bmm_tile_layout_in0_sender_padding.cpp";
    mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid.core_ranges =
        in0_mcast_cores_with_work_and_in_receiver_grid.ranges();
    mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid.compile_time_args = in0_sender_compile_time_args;
    mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid.defines = mm_kernel_in0_sender_writer_defines;
    mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid.config = tt_metal::DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_1,
        .noc = in0_noc,
    };
    mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid.reserve_runtime_args();

    tt_metal::KernelDescriptor* mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid = nullptr;
    tt_metal::KernelDescriptor* mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid = nullptr;
    if (in0_is_sharded) {
        if (in0_mcast_cores_without_work_and_in_receiver_grid.num_cores() > 0) {
            in0_sender_compile_time_args[0] = 0;  // core_has_output_block_work
            in0_sender_compile_time_args[1] = 1;  // core_in_in0_receiver_mcast_grid
            mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid = &program.kernels[num_kernels++];
            mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid->kernel_source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp";
            mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid->core_ranges =
                in0_mcast_cores_without_work_and_in_receiver_grid.ranges();
            mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid->compile_time_args =
                in0_sender_compile_time_args;
            mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid->defines = mm_kernel_in0_sender_writer_defines;
            mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid->config =
                tt_metal::DataMovementConfigDescriptor{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = in0_noc,
                };
            mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid->reserve_runtime_args();
        }
        if (in0_mcast_cores_without_work_and_not_in_receiver_grid.num_cores() > 0) {
            in0_sender_compile_time_args[0] = 0;  // core_has_output_block_work
            in0_sender_compile_time_args[1] = 0;  // core_in_in0_receiver_mcast_grid
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid = &program.kernels[num_kernels++];
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid->kernel_source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp";
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid->core_ranges =
                in0_mcast_cores_without_work_and_not_in_receiver_grid.ranges();
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid->compile_time_args =
                in0_sender_compile_time_args;
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid->defines =
                mm_kernel_in0_sender_writer_defines;
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid->config =
                tt_metal::DataMovementConfigDescriptor{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = in0_noc,
                };
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid->reserve_runtime_args();
        }
    }

    tt_metal::KernelDescriptor* mm_kernel_in0_receiver = nullptr;
    if (!in0_is_sharded and in0_mcast_receivers.num_cores() > 0) {
        mm_kernel_in0_receiver = &program.kernels[num_kernels++];
        mm_kernel_in0_receiver->kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp";
        mm_kernel_in0_receiver->core_ranges = in0_mcast_receivers.ranges();
        mm_kernel_in0_receiver->compile_time_args = {
            // in0 block args
            (std::uint32_t)in0_block_num_tiles,  // in0_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)num_blocks,        // num_blocks
            (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
            (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            // batch args
            (std::uint32_t)B  // batch
        };
        mm_kernel_in0_receiver->config = tt_metal::DataMovementConfigDescriptor{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
        };
        mm_kernel_in0_receiver->reserve_runtime_args();
    }

    auto& mm_kernel_in1_sender_writer = program.kernels[num_kernels++];
    mm_kernel_in1_sender_writer.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_writer_padding.cpp";
    mm_kernel_in1_sender_writer.core_ranges = all_cores_with_work.ranges();
    mm_kernel_in1_sender_writer.defines = mm_kernel_in1_sender_writer_defines;
    mm_kernel_in1_sender_writer.config = tt_metal::DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_0,
        .noc = in1_noc,
    };
    auto& in1_sender_writer_compile_time_args = mm_kernel_in1_sender_writer.compile_time_args;
    in1_sender_writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)in1_is_dram,
        (std::uint32_t)out_is_dram,

        // READER
        // in1 tensor args
        (std::uint32_t)1,                // in1_tensor_stride_w
        (std::uint32_t)N,                // in1_tensor_stride_h
        (std::uint32_t)in0_block_w * N,  // in1_tensor_next_block_stride
        (std::uint32_t)in1_block_w,      // in1_tensor_next_w_dim_block_stride
        // in1 block args
        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,                // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in1 mcast args
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,  // in1_mcast_num_dests
        (std::uint32_t)0,  // in1_mcast_num_cores
        // batch args
        (std::uint32_t)K * N,        // KtNt
        (std::uint32_t)B,            // batch
        (std::uint32_t)bcast_batch,  // bcast_B

        // WRITER
        // out tensor args
        (std::uint32_t)1,                   // out_tensor_stride_w
        (std::uint32_t)N,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,         // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * N,     // out_tensor_next_h_dim_block_stride
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)M * N  // MtNt
    };
    if (bias_buffer != nullptr) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)in3_is_dram);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    } else {
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)fuse_op);
    mm_kernel_in1_sender_writer.reserve_runtime_args();
    // Compute kernel compile time args

    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    tt_metal::KernelDescriptor::CompileTimeArgs compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,        // num_blocks
        out_num_blocks_x,  // out_num_blocks_x
        out_num_blocks_y,  // out_num_blocks_y

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out  // untilize_out
    };

    // Create compute kernel
    // bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    // bool math_approx_mode = false;
    auto& mm_kernel = program.kernels[num_kernels++];
    mm_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
    mm_kernel.core_ranges = all_cores_with_work.ranges();
    mm_kernel.compile_time_args = compute_kernel_args;
    mm_kernel.defines = std::move(mm_kernel_defines);
    mm_kernel.config = tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };
    mm_kernel.reserve_runtime_args();

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = in0_CB_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {tt_metal::CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = in0_tile,
        }},
    });

    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src0_cb_index,
        in0_single_tile_size,
        in0_CB_size / in0_single_tile_size,
        in0_CB_size);
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = in1_CB_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {tt_metal::CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
            .tile = in1_tile,
        }},
    });
    if (in1_is_sharded) {
        program.cbs.back().buffer = in1_buffer;
    }

    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CBHandle cb_src2 = 0;
    if (in0_is_sharded) {
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = in2_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors = {tt_metal::CBFormatDescriptor{
                .buffer_index = src2_cb_index,
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = in0_tile,
            }},
            .buffer = in0_buffer,
        });
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src2_cb_index,
            in0_single_tile_size,
            in2_CB_size / in0_single_tile_size,
            in2_CB_size);

        // Local L1 to store temp vars
        uint32_t l1_cb_index = tt::CBIndex::c_6;
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = 32 * 2,
            .core_ranges = all_cores.ranges(),
            .format_descriptors = {tt_metal::CBFormatDescriptor{
                .buffer_index = l1_cb_index,
                .data_format = tt::DataFormat::Float16_b,
                .page_size = 32 * 2,
            }},
        });
    }

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
        (untilize_out && (in1_num_subblocks > 1))) {
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = out_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors = {tt_metal::CBFormatDescriptor{
                .buffer_index = output_cb_index,
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = output_tile,
            }},
            .buffer = output_is_sharded ? out_buffer : nullptr,
        });
        // interm0
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = interm0_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors = {tt_metal::CBFormatDescriptor{
                .buffer_index = interm0_cb_index,
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = output_tile,
            }}});
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            interm0_cb_index,
            interm0_single_tile_size,
            interm0_CB_size / interm0_single_tile_size,
            interm0_CB_size);
    } else {
        // share buffer
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = out_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {tt_metal::CBFormatDescriptor{
                     .buffer_index = output_cb_index,
                     .data_format = output_data_format,
                     .page_size = output_single_tile_size,
                     .tile = output_tile,
                 },
                 tt_metal::CBFormatDescriptor{
                     .buffer_index = interm0_cb_index,
                     .data_format = interm0_data_format,
                     .page_size = interm0_single_tile_size,
                     .tile = output_tile,
                 }},
            .buffer = output_is_sharded ? out_buffer : nullptr,
        });
    }

    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    tt_metal::CBHandle cb_src3 = 0;
    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = in3_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors = {tt_metal::CBFormatDescriptor{
                .buffer_index = src3_cb_index,
                .data_format = bias_data_format,
                .page_size = bias_single_tile_size,
                .tile = bias_tile,
            }},
            .buffer = bias_is_sharded ? bias_buffer : nullptr,
        });
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src3_cb_index,
            bias_single_tile_size,
            in3_CB_size / bias_single_tile_size,
            in3_CB_size);
    }

    // Parameters for last row, col, or block, no need to re-calc h-dim since there's no split on height
    uint32_t last_per_core_N = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_w = (last_per_core_N - 1) / out_block_w + 1;
    uint32_t last_block_num_nonzero_subblocks_w = (last_out_block_w - 1) / out_subblock_w + 1;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i % num_blocks_x;
        uint32_t output_idx_y = i / num_blocks_x;

        if (in0_is_sharded) {
            tt_metal::KernelDescriptor::CoreRuntimeArgs* mm_in0_sender_args = nullptr;
            if (i < num_cores_with_work) {
                mm_in0_sender_args =
                    &mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid.runtime_args[core.x][core.y];
            } else if (i < in0_mcast_receiver_num_dests) {
                mm_in0_sender_args =
                    &mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid->runtime_args[core.x][core.y];
            } else {
                mm_in0_sender_args =
                    &mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid->runtime_args[core.x][core.y];
            }

            mm_in0_sender_args->push_back(i);
            mm_in0_sender_args->push_back(start_core_noc.x);
            mm_in0_sender_args->push_back(start_core_noc.y);
            mm_in0_sender_args->push_back(end_core_noc.x);
            mm_in0_sender_args->push_back(end_core_noc.y);
            for (auto arg : in0_mcast_noc_x) {
                mm_in0_sender_args->push_back(arg);
            }
            for (auto arg : in0_mcast_noc_y) {
                mm_in0_sender_args->push_back(arg);
            }

            if (fuse_op) {
                fused_op_signaler->push_matmul_fused_op_rt_args(*mm_in0_sender_args, false);
            }
        }
        // in0 sender and in1 sender
        else if (core == start_core) {
            auto& mm_in0_sender_args =
                mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid.runtime_args[core.x][core.y];
            mm_in0_sender_args = {
                // in0 tensor args
                (std::uint32_t)in0_buffer->address(),
                (std::uint32_t)K * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
                // in0 mcast args
                (std::uint32_t)start_core_noc.x,  // in0_mcast_dest_noc_start_x
                (std::uint32_t)start_core_noc.y,  // in0_mcast_dest_noc_start_y
                (std::uint32_t)end_core_noc.x,    // in0_mcast_dest_noc_end_x
                (std::uint32_t)end_core_noc.y,    // in0_mcast_dest_noc_end_y

                // padding args
                (std::uint32_t)out_block_h  // last_block_h
            };

            if (fuse_op) {
                fused_op_signaler->push_matmul_fused_op_rt_args(mm_in0_sender_args, false);
            }
        }
        // in0 receiver and in 1 sender
        else {
            mm_kernel_in0_receiver->runtime_args[core.x][core.y] = {
                // in0 mcast args
                (std::uint32_t)top_left_core_physical.x,  // in0_mcast_sender_noc_x
                (std::uint32_t)top_left_core_physical.y   // in0_mcast_sender_noc_y
            };  // RISCV_1_default
        }
        if (i < num_cores_with_work) {
            auto& mm_in1_sender_writer_args = mm_kernel_in1_sender_writer.runtime_args[core.x][core.y];
            mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)in1_buffer->address(),
                (std::uint32_t)per_core_N * output_idx_x,  // in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)0,  // in1_mcast_dest_noc_start_x
                (std::uint32_t)0,  // in1_mcast_dest_noc_start_y
                (std::uint32_t)0,  // in1_mcast_dest_noc_end_x
                (std::uint32_t)0,  // in1_mcast_dest_noc_end_y

                // WRITER
                // out tensor args
                (std::uint32_t)out_buffer->address(),
                (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * N  // out_tensor_start_tile_id
            };

            if (output_idx_x == num_blocks_x - 1) {
                // padding args (READER)
                mm_in1_sender_writer_args.push_back(last_out_block_w);

                // padding args (WRITER)
                mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                mm_in1_sender_writer_args.push_back(out_subblock_h);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);  // out_num_nonzero_subblocks_w
                mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
                mm_in1_sender_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                mm_in1_sender_writer_args.push_back(last_block_padded_block_tiles_w_skip);
            } else {
                // padding args (READER)
                mm_in1_sender_writer_args.push_back(out_block_w);

                // padding args (WRITER)
                mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                mm_in1_sender_writer_args.push_back(out_subblock_h);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);  // out_num_nonzero_subblocks_w
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                mm_in1_sender_writer_args.push_back(out_subblock_w);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }

            if (bias_buffer != nullptr) {
                mm_in1_sender_writer_args.push_back((std::uint32_t)bias_buffer->address());
                mm_in1_sender_writer_args.push_back(
                    (std::uint32_t)per_core_N * output_idx_x);  // in3_tensor_start_tile_id
            } else {                                            // Placeholder args
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }
            if (!output_is_sharded) {
                if (output_idx_x == num_blocks_x - 1) {
                    mm_in1_sender_writer_args.push_back(last_out_num_blocks_w);
                } else {
                    mm_in1_sender_writer_args.push_back(out_num_blocks_x);
                }
            }

            if (fuse_op) {
                fused_op_signaler->push_matmul_fused_op_rt_args(mm_in1_sender_writer_args, true);
            }
        }
    }

    program.kernels.resize(num_kernels);
    return program;
}

tt::tt_metal::ProgramDescriptor create_program_mcast_in1(
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    CoreCoord compute_with_storage_grid_size,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* bias_buffer,
    tt_metal::Buffer* out_buffer,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool in0_is_sharded,
    bool output_is_sharded,
    bool untilize_out) {
    // currently only support transpose of the full tile
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    tt_metal::ProgramDescriptor program;

    bool fuse_op = false;

    uint32_t num_blocks = K / in0_block_w;
    // Only enable packer l1 accumulation when there are num_blocks > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    // For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
    bool packer_l1_acc_en = packer_l1_acc && (((bias_buffer != nullptr) && num_blocks > 1) || (num_blocks > 2));

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (in0_is_sharded) {
        in0_CB_tiles = num_blocks * per_core_M * in0_block_w * B;
    } else if (B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2;  // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    bool extract_shard_sub_blocks = false;
    uint32_t in0_shard_height_in_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0];
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
        // NOTE: Criteria for extract_shard_sub_blocks is different from mcast in0
        // In the reader kernel, always need to copy to cb0 even for height=1 shards since we may not always do mcast
        // In mcast in0 sharded reader kernel, this is handled by mcast with loopback src
        // For mcast in1, if we don't need to extract_shard_sub_blocks, set the sharded in0 cb to cb0
        // For mcast in0, sharded in0 cb is always cb2
        if (in0_shard_width_in_tiles / in0_block_w > 1) {
            extract_shard_sub_blocks = true;
        }
    }
    uint32_t in2_CB_tiles = in0_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 2;  // double buffer
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_shard_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    if (output_is_sharded) {
        out_CB_tiles = out_shard_tiles;
    }
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    CoreCoord start_core = {0, 0};
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;

    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores = num_blocks_total;

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        tt::tt_metal::num_cores_to_corerangeset(start_core, num_cores, compute_with_storage_grid_size, row_major);
    CoreRange in1_mcast_receiver_cores_bounding_box = all_cores.bounding_box();
    uint32_t in1_mcast_receiver_num_cores = in1_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid

    CoreRange in1_mcast_sender(start_core, start_core);
    CoreRangeSet in1_mcast_receivers;
    if (in1_mcast_receiver_num_cores > 1) {
        auto receiver_start_core = start_core.x != (compute_with_storage_grid_size.x - 1)
                                       ? CoreCoord{start_core.x + 1, start_core.y}
                                       : CoreCoord{start_core.x, start_core.y + 1};
        in1_mcast_receivers = tt::tt_metal::num_cores_to_corerangeset(
            receiver_start_core, num_cores - 1, compute_with_storage_grid_size, row_major);
    }

    // Mcast args
    uint32_t in1_mcast_sender_semaphore_id = program.add_semaphore(all_cores.ranges(), INVALID);
    uint32_t in1_mcast_receiver_semaphore_id = program.add_semaphore(all_cores.ranges(), INVALID);

    CoreCoord top_left_core = in1_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in1_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    bool in3_is_dram = true;
    if (bias_buffer != nullptr) {
        in3_is_dram = bias_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    }
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;

    tt_metal::KernelDescriptor::Defines mm_kernel_defines;
    tt_metal::KernelDescriptor::Defines mm_kernel_in0_sender_defines;
    tt_metal::KernelDescriptor::Defines mm_kernel_in1_sender_writer_defines;
    tt_metal::KernelDescriptor::Defines mm_kernel_in1_receiver_writer_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines.emplace_back("FUSE_BIAS", "1");
        mm_kernel_in1_sender_writer_defines.emplace_back("FUSE_BIAS", "1");
        mm_kernel_in1_receiver_writer_defines.emplace_back("FUSE_BIAS", "1");
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines.emplace_back("PACK_RELU", "1");
        } else {
            using ttnn::operations::unary::utils::get_defines_vec;
            auto extra_defines =
                get_defines_vec(fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i");
            mm_kernel_defines.insert(mm_kernel_defines.end(), extra_defines.begin(), extra_defines.end());
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines.emplace_back("PACKER_L1_ACC", "1");
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    if (in1_transpose_tile) {
        mm_kernel_defines.emplace_back("IN1_TRANSPOSE_TILE", "1");
    }

    bmm_op_utils::add_stagger_defines_if_needed(device->arch(), num_cores, mm_kernel_defines);

    if (in0_is_sharded) {
        mm_kernel_in0_sender_defines.emplace_back("IN0_SHARDED", "1");
    }
    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines.emplace_back("OUT_SHARDED", "1");
        mm_kernel_in1_receiver_writer_defines.emplace_back("OUT_SHARDED", "1");
    }

    mm_kernel_in0_sender_defines.emplace_back("SKIP_MCAST", "1");

    if (in1_mcast_receiver_num_cores == 1) {
        mm_kernel_in1_sender_writer_defines.emplace_back("SKIP_MCAST", "1");
    }

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());

    constexpr auto max_num_kernels = 4;
    program.kernels.resize(max_num_kernels);
    size_t num_kernels = 0;

    auto& mm_kernel_in0_sender = program.kernels[num_kernels++];
    mm_kernel_in0_sender.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp";
    mm_kernel_in0_sender.core_ranges = all_cores.ranges();
    mm_kernel_in0_sender.defines = mm_kernel_in0_sender_defines;
    mm_kernel_in0_sender.config = tt_metal::DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_1,
        .noc = in0_noc,
    };
    mm_kernel_in0_sender.compile_time_args = {// interleaved accessor args
                                              (std::uint32_t)in0_is_dram,

                                              // in0 tensor args
                                              (std::uint32_t)1,                // in0_tensor_stride_w
                                              (std::uint32_t)K,                // in0_tensor_stride_h
                                              (std::uint32_t)in0_block_w,      // in0_tensor_next_block_stride
                                              (std::uint32_t)K * in0_block_h,  // in0_tensor_next_h_dim_block_stride
                                                                               // in0 block args
                                              (std::uint32_t)in0_block_w,      // in0_block_w
                                              (std::uint32_t)in0_block_h,      // in0_block_h
                                              (std::uint32_t)in0_block_w * in0_block_h,  // in0_block_num_tiles
                                              (std::uint32_t)extract_shard_sub_blocks,
                                              (std::uint32_t)in0_shard_width_in_tiles,
                                              (std::uint32_t)in0_shard_height_in_tiles,
                                              // in0/in1 common args
                                              (std::uint32_t)num_blocks,        // num_blocks
                                              (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
                                              (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
                                                                                // in0 mcast args
                                              (std::uint32_t)0,
                                              (std::uint32_t)0,
                                              (std::uint32_t)0,      // in0_mcast_num_dests
                                              (std::uint32_t)0,      // in0_mcast_num_cores
                                                                     // batch args
                                              (std::uint32_t)M * K,  // MtKt
                                              (std::uint32_t)B,      // batch
                                              (std::uint32_t)fuse_op};
    mm_kernel_in0_sender.reserve_runtime_args();

    auto& mm_kernel_in1_sender_writer = program.kernels[num_kernels++];
    mm_kernel_in1_sender_writer.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_writer_padding.cpp";
    mm_kernel_in1_sender_writer.core_ranges = {in1_mcast_sender};
    mm_kernel_in1_sender_writer.defines = mm_kernel_in1_sender_writer_defines;
    mm_kernel_in1_sender_writer.config = tt_metal::DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_0,
        .noc = in1_noc,
    };
    mm_kernel_in1_sender_writer.compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)in1_is_dram,
        (std::uint32_t)out_is_dram,

        // READER
        // in1 tensor args
        (std::uint32_t)1,                // in1_tensor_stride_w
        (std::uint32_t)N,                // in1_tensor_stride_h
        (std::uint32_t)in0_block_w * N,  // in1_tensor_next_block_stride
        (std::uint32_t)in1_block_w,      // in1_tensor_next_w_dim_block_stride
        // in1 block args
        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,                // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in1 mcast args
        (std::uint32_t)in1_mcast_sender_semaphore_id,
        (std::uint32_t)in1_mcast_receiver_semaphore_id,
        (std::uint32_t)num_cores - 1,                     // in1_mcast_num_dests
        (std::uint32_t)in1_mcast_receiver_num_cores - 1,  // in1_mcast_num_cores
        // batch args
        (std::uint32_t)K * N,        // KtNt
        (std::uint32_t)B,            // batch
        (std::uint32_t)bcast_batch,  // bcast_B

        // WRITER
        // out tensor args
        (std::uint32_t)1,                   // out_tensor_stride_w
        (std::uint32_t)N,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,         // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * N,     // out_tensor_next_h_dim_block_stride
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)M * N  // MtNt
    };
    if (bias_buffer != nullptr) {
        mm_kernel_in1_sender_writer.compile_time_args.push_back((std::uint32_t)in3_is_dram);
        mm_kernel_in1_sender_writer.compile_time_args.push_back((std::uint32_t)1);
    } else {
        mm_kernel_in1_sender_writer.compile_time_args.push_back(0);  // Placeholder; not used
        mm_kernel_in1_sender_writer.compile_time_args.push_back(0);  // Placeholder; not used
    }
    mm_kernel_in1_sender_writer.compile_time_args.push_back((std::uint32_t)fuse_op);
    mm_kernel_in1_sender_writer.reserve_runtime_args();

    tt::tt_metal::KernelDescriptor* mm_kernel_in1_receiver_writer = nullptr;
    if (in1_mcast_receivers.num_cores() > 0) {
        mm_kernel_in1_receiver_writer = &program.kernels[num_kernels++];
        mm_kernel_in1_receiver_writer->kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp";
        mm_kernel_in1_receiver_writer->core_ranges = in1_mcast_receivers.ranges();
        mm_kernel_in1_receiver_writer->defines = mm_kernel_in1_receiver_writer_defines;
        mm_kernel_in1_receiver_writer->config = tt_metal::DataMovementConfigDescriptor{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
        };
        mm_kernel_in1_receiver_writer->compile_time_args = {
            // interleaved accessor args
            (std::uint32_t)out_is_dram,

            // READER
            // in1 block args
            (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)num_blocks,        // num_blocks
            (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
            (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
            // in1 mcast args
            (std::uint32_t)in1_mcast_sender_semaphore_id,
            (std::uint32_t)in1_mcast_receiver_semaphore_id,
            // batch args
            (std::uint32_t)B,  // batch

            // WRITER
            // out tensor args
            (std::uint32_t)1,                   // out_tensor_stride_w
            (std::uint32_t)N,                   // out_tensor_stride_h
            (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
            (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
            (std::uint32_t)out_block_w,         // out_tensor_next_w_dim_block_stride
            (std::uint32_t)out_block_h * N,     // out_tensor_next_h_dim_block_stride
            // out subblock args
            (std::uint32_t)out_subblock_w,                     // out_subblock_w
            (std::uint32_t)out_subblock_h,                     // out_subblock_h
            (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
            // batch args
            (std::uint32_t)M * N  // MtNt
        };
        if (bias_buffer != nullptr) {
            mm_kernel_in1_receiver_writer->compile_time_args.push_back((std::uint32_t)in1_block_w);
        }
        mm_kernel_in1_receiver_writer->reserve_runtime_args();
    }

    // Compute kernel compile time args

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    // Create compute kernel
    // bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    // bool math_approx_mode = false;
    auto& mm_kernel = program.kernels[num_kernels++];
    mm_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
    mm_kernel.core_ranges = all_cores.ranges();
    mm_kernel.compile_time_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,        // num_blocks
        out_num_blocks_x,  // out_num_blocks_x
        out_num_blocks_y,  // out_num_blocks_y

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out  // untilize_out
    };
    mm_kernel.defines = mm_kernel_defines;
    mm_kernel.config = tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };
    mm_kernel.reserve_runtime_args();

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = in0_CB_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors =
            {
                tt_metal::CBFormatDescriptor{
                    .buffer_index = src0_cb_index,
                    .data_format = in0_data_format,
                    .page_size = in0_single_tile_size,
                    .tile = in0_tile,
                },
            },
        .buffer = in0_is_sharded && !extract_shard_sub_blocks ? in0_buffer : nullptr,
    });
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src0_cb_index,
        in0_single_tile_size,
        in0_CB_size / in0_single_tile_size,
        in0_CB_size);

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CBHandle cb_src2 = 0;
    if (in0_is_sharded and extract_shard_sub_blocks) {  // in0_is_sharded is technically redundant
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = in2_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = src2_cb_index,
                        .data_format = in0_data_format,
                        .page_size = in0_single_tile_size,
                        .tile = in0_tile,
                    },
                },
            .buffer = in0_buffer,
        });
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src2_cb_index,
            in0_single_tile_size,
            in2_CB_size / in0_single_tile_size,
            in2_CB_size);
    }

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = in1_CB_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors =
            {
                tt_metal::CBFormatDescriptor{
                    .buffer_index = src1_cb_index,
                    .data_format = in1_data_format,
                    .page_size = in1_single_tile_size,
                    .tile = in1_tile,
                },
            },
    });
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;

    if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
        (untilize_out && (in1_num_subblocks > 1))) {
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = out_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = output_cb_index,
                        .data_format = output_data_format,
                        .page_size = output_single_tile_size,
                        .tile = output_tile,
                    },
                },
            .buffer = output_is_sharded ? out_buffer : nullptr,
        });
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = interm0_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = interm0_cb_index,
                        .data_format = interm0_data_format,
                        .page_size = interm0_single_tile_size,
                        .tile = output_tile,
                    },
                },
        });
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            interm0_cb_index,
            interm0_single_tile_size,
            interm0_CB_size / interm0_single_tile_size,
            interm0_CB_size);
    } else {
        // share buffer
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = out_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = output_cb_index,
                        .data_format = output_data_format,
                        .page_size = output_single_tile_size,
                        .tile = output_tile,
                    },
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = interm0_cb_index,
                        .data_format = interm0_data_format,
                        .page_size = interm0_single_tile_size,
                        .tile = output_tile,
                    },
                },
            .buffer = output_is_sharded ? out_buffer : nullptr,
        });
    }

    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = in3_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = src3_cb_index,
                        .data_format = bias_data_format,
                        .page_size = bias_single_tile_size,
                        .tile = bias_tile,
                    },
                },
        });
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src3_cb_index,
            bias_single_tile_size,
            in3_CB_size / bias_single_tile_size,
            in3_CB_size);
    }

    // Parameters for last row, col, or block
    uint32_t last_per_core_M = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_out_block_h = last_per_core_M % out_block_h == 0 ? out_block_h : last_per_core_M % out_block_h;
    uint32_t last_out_num_blocks_h = (last_per_core_M - 1) / out_block_h + 1;
    uint32_t last_block_num_nonzero_subblocks_h = (last_out_block_h - 1) / out_subblock_h + 1;
    uint32_t last_subblock_of_last_block_h =
        last_out_block_h % out_subblock_h == 0 ? out_subblock_h : last_out_block_h % out_subblock_h;
    uint32_t last_block_padded_block_tiles_h_skip =
        (out_block_h / out_subblock_h - last_block_num_nonzero_subblocks_h) * (out_block_w * out_subblock_h);

    CoreCoord start_core_noc = bottom_right_core_physical;
    CoreCoord end_core_noc = top_left_core_physical;
    if (in1_noc == tt::tt_metal::NOC::NOC_0) {
        std::swap(start_core_noc, end_core_noc);
    }

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i / num_blocks_y;
        uint32_t output_idx_y = i % num_blocks_y;

        // in0 sender and in1 sender
        if (core == start_core) {
            auto& mm_in1_sender_writer_args = mm_kernel_in1_sender_writer.runtime_args[core.x][core.y];
            mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)in1_buffer->address(),
                (std::uint32_t)per_core_N * output_idx_x,  // in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)start_core_noc.x,  // in1_mcast_dest_noc_start_x
                (std::uint32_t)start_core_noc.y,  // in1_mcast_dest_noc_start_y
                (std::uint32_t)end_core_noc.x,    // in1_mcast_dest_noc_end_x
                (std::uint32_t)end_core_noc.y,    // in1_mcast_dest_noc_end_y

                // WRITER
                // out tensor args
                (std::uint32_t)out_buffer->address(),
                (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * N,  // out_tensor_start_tile_id

                // padding args (READER)
                (std::uint32_t)out_block_w,  // last_block_w
                // padding args (WRITER)
                (std::uint32_t)out_block_h / out_subblock_h,
                (std::uint32_t)out_subblock_h,
                (std::uint32_t)0,
                (std::uint32_t)out_block_w / out_subblock_w,
                (std::uint32_t)out_block_w / out_subblock_w,
                (std::uint32_t)out_subblock_w,
                (std::uint32_t)0,
                (std::uint32_t)0};

            if (bias_buffer != nullptr) {
                mm_in1_sender_writer_args.push_back((std::uint32_t)bias_buffer->address());
                mm_in1_sender_writer_args.push_back(
                    (std::uint32_t)per_core_N * output_idx_x);  // in3_tensor_start_tile_id
            } else {
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }
            if (!output_is_sharded) {
                mm_in1_sender_writer_args.push_back(out_num_blocks_x);
            }
        }
        // in0 sender and in1 receiver
        else {
            auto& mm_in1_receiver_writer_args = mm_kernel_in1_receiver_writer->runtime_args[core.x][core.y];
            mm_in1_receiver_writer_args = {
                // READER
                // in1 mcast args
                (std::uint32_t)top_left_core_physical.x,  // in1_mcast_sender_noc_x
                (std::uint32_t)top_left_core_physical.y,  // in1_mcast_sender_noc_y

                // WRITER
                // out tensor args
                (std::uint32_t)out_buffer->address(),                                     // out_tensor_addr
                (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * N  // out_tensor_start_tile_id
            };

            if (output_idx_y == num_blocks_y - 1) {
                // padding args (WRITER)
                mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                mm_in1_receiver_writer_args.push_back(out_subblock_w);
                mm_in1_receiver_writer_args.push_back(0);
                mm_in1_receiver_writer_args.push_back(0);
            } else {
                // padding args (WRITER)
                mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                mm_in1_receiver_writer_args.push_back(out_subblock_h);
                mm_in1_receiver_writer_args.push_back(0);
                mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                mm_in1_receiver_writer_args.push_back(out_subblock_w);
                mm_in1_receiver_writer_args.push_back(0);
                mm_in1_receiver_writer_args.push_back(0);
            }
            if (!output_is_sharded) {
                if (output_idx_y == num_blocks_y - 1) {
                    mm_in1_receiver_writer_args.push_back(last_out_num_blocks_h);
                    mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                } else {
                    mm_in1_receiver_writer_args.push_back(out_num_blocks_y);
                    mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                }
            }
        }

        auto& mm_in0_sender_args = mm_kernel_in0_sender.runtime_args[core.x][core.y];
        mm_in0_sender_args = {
            // in0 tensor args
            (std::uint32_t)in0_buffer->address(),
            (std::uint32_t)K * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
            // in0 mcast args
            (std::uint32_t)0,  // in0_mcast_dest_noc_start_x
            (std::uint32_t)0,  // in0_mcast_dest_noc_start_y
            (std::uint32_t)0,  // in0_mcast_dest_noc_end_x
            (std::uint32_t)0,  // in0_mcast_dest_noc_end_y

            // padding args
            (std::uint32_t)per_core_M  // last_block_h
        };
    }
    program.kernels.resize(num_kernels);
    return program;
}

enum class CORE_TYPE : uint32_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

tt::tt_metal::ProgramDescriptor create_program_gather_in0(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    bool dst_full_sync_en,
    CoreCoord compute_with_storage_grid_size,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const CoreRangeSet& hop_cores,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* out_buffer,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    const bool in1_is_dram_interleaved = in1_buffer->is_dram() && !b.is_sharded();

    tt_metal::ProgramDescriptor program;

    /* Core setup */
    constexpr bool row_major = true;
    CoreRangeSet all_worker_cores = a.shard_spec().value().grid;
    CoreRangeSet non_idle_cores = all_worker_cores.merge(hop_cores);
    auto subdevice_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
    std::vector<CoreRange> non_idle_cores_vec;
    for (auto& cr : subdevice_cores.ranges()) {
        auto intersection = non_idle_cores.intersection(cr);
        if (intersection.size() > 0) {
            non_idle_cores_vec.push_back(intersection.bounding_box());
        }
    }
    CoreRangeSet all_cores = CoreRangeSet(non_idle_cores_vec);
    auto ring_list = all_worker_cores.ranges();
    const auto& hop_list = hop_cores.ranges();
    ring_list.insert(ring_list.end(), hop_list.begin(), hop_list.end());

    CoreRangeSet ring_cores = CoreRangeSet(ring_list);
    const uint32_t num_cores = all_worker_cores.num_cores();
    const uint32_t ring_size = num_cores;

    uint32_t num_hop_cores = hop_cores.num_cores();
    bool use_hop_cores = num_hop_cores > 0;

    /* Inner dim padding */
    const uint32_t Kt_pad = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1] * num_cores;
    in0_block_w = Kt_pad / num_cores;

    uint32_t num_blocks = Kt_pad / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    bool use_global_cb = global_cb.has_value();

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    /* in0 */
    uint32_t in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    uint32_t in0_CB_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    /* in1 */
    uint32_t in1_shard_height_in_tiles = 0;
    uint32_t in1_shard_width_in_tiles = 0;
    uint32_t in1_CB_tiles = 0;
    uint32_t in1_tensor_width_in_tiles = b.get_padded_shape()[-1] / in1_tile.get_tile_shape()[1];

    if (!in1_is_dram_interleaved) {
        in1_shard_height_in_tiles = in1_buffer->shard_spec().shape()[0] / in1_tile.get_tile_shape()[0];
        in1_shard_width_in_tiles =
            in1_buffer->shard_spec().shape()[1] / in1_tile.get_tile_shape()[1] / num_global_cb_receivers;
        in1_CB_tiles = in1_shard_height_in_tiles * in1_shard_width_in_tiles;
    } else {
        in1_CB_tiles = 2 * in0_shard_width_in_tiles * per_core_N;  // Double buffered
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    /* in2 */
    uint32_t in2_single_tile_size = in0_single_tile_size;
    uint32_t in2_CB_tiles = (ring_size - 1) * in0_CB_tiles;  // All shards except local
    uint32_t in2_CB_size = in2_CB_tiles * in2_single_tile_size;

    /* out */
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t K_ = K;
    std::vector<uint32_t> unpadded_in0_shard_widths_in_tiles(num_cores, 0);
    for (uint32_t i = 0; i < num_cores && K_ > 0; ++i) {
        unpadded_in0_shard_widths_in_tiles[i] = std::min(K_, in0_shard_width_in_tiles);
        K_ -= unpadded_in0_shard_widths_in_tiles[i];
    }

    /* semaphores */
    uint32_t in0_signal_semaphore_id = program.add_semaphore(all_cores.ranges(), INVALID);

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in1_block_height_in_tiles = in0_block_w;
    uint32_t in1_block_num_tiles = out_subblock_w * in1_block_height_in_tiles * in1_num_subblocks;
    uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size;
    uint32_t in1_tensor_size_bytes = in1_block_num_tiles * num_blocks * in1_single_tile_size;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    /* Kernel defines */
    tt_metal::KernelDescriptor::Defines mm_in1_kernel_defines;
    tt_metal::KernelDescriptor::Defines mm_kernel_defines;

    if (use_global_cb) {
        mm_in1_kernel_defines.emplace_back("ENABLE_GLOBAL_CB", "1");
        mm_kernel_defines.emplace_back("ENABLE_GLOBAL_CB", "1");
    }

    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines.emplace_back("PACK_RELU", "1");
        } else {
            using ttnn::operations::unary::utils::get_defines_vec;
            auto extra_defines =
                get_defines_vec(fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i");
            mm_kernel_defines.insert(mm_kernel_defines.end(), extra_defines.begin(), extra_defines.end());
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines.emplace_back("PACKER_L1_ACC", "1");
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    bmm_op_utils::add_stagger_defines_if_needed(device->arch(), num_cores, mm_kernel_defines);

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());

    bool use_dedicated_noc = true;
    tt_metal::NOC_MODE noc_mode =
        use_dedicated_noc ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC;

    /* Create the kernels */
    constexpr auto max_num_kernels = 3;
    program.kernels.resize(max_num_kernels);
    size_t num_kernels = 0;

    auto& mm_kernel_in0 = program.kernels[num_kernels++];
    mm_kernel_in0.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp";
    mm_kernel_in0.core_ranges = all_cores.ranges();
    mm_kernel_in0.compile_time_args = {
        (std::uint32_t)in0_shard_width_in_tiles,
        (std::uint32_t)per_core_M,  // in0_shard_height_in_tiles
        (std::uint32_t)B,           // batch
        (std::uint32_t)ring_size,   // ring_size
        (std::uint32_t)in0_signal_semaphore_id,
    };
    mm_kernel_in0.config = tt_metal::DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_1,
        .noc = in0_noc,
        .noc_mode = noc_mode,
    };
    mm_kernel_in0.reserve_runtime_args();

    auto& mm_kernel_in1_sender_writer = program.kernels[num_kernels++];
    mm_kernel_in1_sender_writer.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp";
    mm_kernel_in1_sender_writer.core_ranges = all_cores.ranges();
    mm_kernel_in1_sender_writer.compile_time_args = {
        (std::uint32_t)in1_is_dram_interleaved,    // in1_is_dram_interleaved
        (std::uint32_t)in1_block_height_in_tiles,  // in1_block_height_in_tiles
        (std::uint32_t)per_core_N,                 // in1_block_width_in_tiles
        (std::uint32_t)in1_tensor_width_in_tiles,  // in1_tensor_width_in_tiles
        (std::uint32_t)num_blocks,                 // num_blocks
        (std::uint32_t)B,                          // batch
    };
    mm_kernel_in1_sender_writer.defines = mm_in1_kernel_defines;
    mm_kernel_in1_sender_writer.config = tt_metal::DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_0,
        .noc = in1_noc,
        .noc_mode = noc_mode,
    };
    mm_kernel_in1_sender_writer.reserve_runtime_args();

    auto& mm_kernel = program.kernels[num_kernels++];
    mm_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
        "bmm_large_block_zm_fused_bias_activation_gathered.cpp";
    mm_kernel.core_ranges = all_cores.ranges();
    mm_kernel.compile_time_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,      // in1_num_subblocks
        in1_block_num_tiles,    // in1_block_num_tiles
        in1_block_size_bytes,   // in1_block_size_bytes
        in1_tensor_size_bytes,  // in1_tensor_size_bytes
        in1_per_core_w,         // in1_per_core_w

        num_blocks,  // num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out,             // untilize_out
        in1_is_dram_interleaved,  // in1_is_dram_interleaved
    };
    mm_kernel.defines = std::move(mm_kernel_defines);
    mm_kernel.config = tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };
    mm_kernel.reserve_runtime_args();

    /* Create circular buffers */
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = in0_CB_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors =
            {
                tt_metal::CBFormatDescriptor{
                    .buffer_index = src0_cb_index,
                    .data_format = in0_data_format,
                    .page_size = in0_single_tile_size,
                    .tile = in0_tile,
                },
            },
        .buffer = in0_buffer,
    });

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CBHandle cb_src1;
    if (use_global_cb) {
        uint32_t in1_block_size_bytes = in1_single_tile_size * in1_block_num_tiles;
        uint32_t remote_cb_index = tt::CBIndex::c_31;
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = (global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = src1_cb_index,
                        .data_format = in1_data_format,
                        .page_size = in1_single_tile_size,
                    },
                },
            .remote_format_descriptors = {tt_metal::CBFormatDescriptor{
                .buffer_index = remote_cb_index,
                .data_format = in1_data_format,
                .page_size = in1_block_size_bytes,
            }},
            .global_circular_buffer = &*global_cb,
        });
    } else {
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = in1_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = src1_cb_index,
                        .data_format = in1_data_format,
                        .page_size = in1_single_tile_size,
                        .tile = in1_tile,
                    },
                },
            .buffer = in1_is_dram_interleaved ? nullptr : in1_buffer,
        });
    }

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = in2_CB_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors =
            {
                tt_metal::CBFormatDescriptor{
                    .buffer_index = src2_cb_index,
                    .data_format = in0_data_format,
                    .page_size = in2_single_tile_size,
                    .tile = in0_tile,
                },
            },
    });

    uint32_t sync_cb_index = tt::CBIndex::c_3;
    uint32_t sync_cb_size_bytes = 16;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = sync_cb_size_bytes,
        .core_ranges = all_cores.ranges(),
        .format_descriptors =
            {
                tt_metal::CBFormatDescriptor{
                    .buffer_index = sync_cb_index,
                    .data_format = tt::DataFormat::UInt16,
                    .page_size = sync_cb_size_bytes,
                },
            },
    });

    uint32_t sync_cb2_index = tt::CBIndex::c_4;
    uint32_t sync_cb2_size_bytes = 16;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = sync_cb2_size_bytes,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {
            tt_metal::CBFormatDescriptor{
                .buffer_index = sync_cb2_index,
                .data_format = tt::DataFormat::UInt16,
                .page_size = sync_cb2_size_bytes,
            },
        }});

    uint32_t output_cb_index = tt::CBIndex::c_5;  // output operands start at index 16
    uint32_t interm0_cb_index = tt::CBIndex::c_6;

    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = out_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = output_cb_index,
                        .data_format = output_data_format,
                        .page_size = output_single_tile_size,
                        .tile = output_tile,
                    },
                },
            .buffer = out_buffer,
        });

        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = interm0_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = interm0_cb_index,
                        .data_format = interm0_data_format,
                        .page_size = interm0_single_tile_size,
                        .tile = output_tile,
                    },
                },
        });
    } else {
        // share buffer
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = out_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = output_cb_index,
                        .data_format = output_data_format,
                        .page_size = output_single_tile_size,
                        .tile = output_tile,
                    },
                    tt_metal::CBFormatDescriptor{
                        .buffer_index = interm0_cb_index,
                        .data_format = interm0_data_format,
                        .page_size = interm0_single_tile_size,
                        .tile = output_tile,
                    },
                },
            .buffer = out_buffer,
        });
    }

    // for all the cores in the rect grid, we send one rt arg to determine if they are worker core
    auto all_cores_vec = corerange_to_cores(all_cores, std::nullopt, row_major);
    auto worker_cores_vec = corerange_to_cores(all_worker_cores, std::nullopt, row_major);
    auto hop_cores_vec = corerange_to_cores(hop_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < all_cores_vec.size(); ++i) {
        auto core = all_cores_vec[i];

        auto all_worker_cores_iter = std::find(worker_cores_vec.begin(), worker_cores_vec.end(), core);
        auto hop_cores_iter = std::find(hop_cores_vec.begin(), hop_cores_vec.end(), core);
        bool core_is_in_all_worker_cores = all_worker_cores_iter != worker_cores_vec.end();
        bool core_is_in_hop_cores = hop_cores_iter != hop_cores_vec.end();
        if (!use_hop_cores) {
            core_is_in_hop_cores = false;
        }

        if (!core_is_in_all_worker_cores && !core_is_in_hop_cores) {  // not worker core and not hop core
            auto core_type = CORE_TYPE::IDLE_CORE;                    // idle core
            // in0
            mm_kernel_in0.runtime_args[core.x][core.y] = {(std::uint32_t)core_type};

            // in1
            mm_kernel_in1_sender_writer.runtime_args[core.x][core.y] = {(std::uint32_t)core_type};

            // compute
            mm_kernel.runtime_args[core.x][core.y] = {(std::uint32_t)core_type};
        }
    }

    /* Runtime args */
    for (uint32_t i = 0; i < num_cores; ++i) {
        bool send_to_hop_core = i == 0 && use_hop_cores;
        const auto& core = worker_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        /* in0 */
        auto core_type = CORE_TYPE::WORKER_CORE;  // worker core
        CoreCoord next_core;
        if (send_to_hop_core) {
            next_core = hop_cores_vec[0];  // Send to first hop core
        } else {
            uint32_t next_i = i == 0 ? num_cores - 1 : i - 1;
            next_core = worker_cores_vec[next_i % num_cores];
        }
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
        uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

        auto& mm_in0_args = mm_kernel_in0.runtime_args[core.x][core.y];
        mm_in0_args = {
            (std::uint32_t)core_type,
            i,                // ring_index
            next_core_noc.x,  // next_core_noc_x
            next_core_noc.y,  // next_core_noc_y
            noc,
            (std::uint32_t)false,  // end_of_hop
        };

        for (auto arg : unpadded_in0_shard_widths_in_tiles) {
            mm_in0_args.push_back(arg);
        }

        /* in1 */
        auto& mm_in1_args = mm_kernel_in1_sender_writer.runtime_args[core.x][core.y];
        mm_in1_args = {
            (std::uint32_t)core_type,
            in1_buffer->address(),  // in1_tensor_addr
            i,                      // ring_idx
        };

        /* compute */
        auto& mm_kernel_compute_args = mm_kernel.runtime_args[core.x][core.y];
        mm_kernel_compute_args = {
            (std::uint32_t)core_type,
            i,  // ring_idx
        };
        for (auto arg : unpadded_in0_shard_widths_in_tiles) {
            mm_kernel_compute_args.push_back(arg);
        }
    }

    // Runtime args for hop cores
    for (uint32_t i = 0; i < num_hop_cores; ++i) {
        bool end_of_hop = i == num_hop_cores - 1;

        auto core_type = CORE_TYPE::HOP_CORE;  // hop core
        const auto& core = hop_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        /* in0 */
        CoreCoord next_core = end_of_hop ? worker_cores_vec[num_cores - 1] : hop_cores_vec[i + 1];
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
        uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

        mm_kernel_in0.runtime_args[core.x][core.y] = {
            (std::uint32_t)core_type,
            0,                // ring_index
            next_core_noc.x,  // next_core_noc_x
            next_core_noc.y,  // next_core_noc_y
            noc,
            (std::uint32_t)end_of_hop,  // end_of_hop
        };

        // in1
        mm_kernel_in1_sender_writer.runtime_args[core.x][core.y] = {
            (std::uint32_t)core_type,
        };

        // compute
        mm_kernel.runtime_args[core.x][core.y] = {
            (std::uint32_t)core_type,
        };
    }

    program.kernels.resize(num_kernels);
    return program;
}

}  // namespace reuse_mcast_1d_optimized_helpers

namespace ttnn {

namespace operations {

namespace matmul {

tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_mcast_1d_optimized_(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output,
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
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    const auto &ashape = a.get_padded_shape(), bshape = b.get_padded_shape();
    auto in0_tile = a.get_tensor_spec().tile();
    auto in1_tile = b.get_tensor_spec().tile();
    // cannot use the output tensor tile directly as that might be changed by user override
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile_shape[0], in1_tile_shape[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());          // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());          // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());  // output

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;  // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        auto& c = bias.value();
        TT_FATAL(c.storage_type() == StorageType::DEVICE, "Error");
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

        bias_buffer = c.buffer();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.get_dtype());
    }

    tt_metal::IDevice* device = a.device();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    TT_FATAL(in0_buffer->size() % in0_single_tile_size == 0, "Error");
    TT_FATAL(in1_buffer->size() % in1_single_tile_size == 0, "Error");

    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] and B.shape[-2]) must match for A and B in bmm_op");  // A.K == B.K
    TT_FATAL(ashape[-2] % in0_tile_shape[0] == 0, "Error");
    TT_FATAL(ashape[-1] % in0_tile_shape[1] == 0, "Error");
    TT_FATAL(bshape[-2] % in1_tile_shape[0] == 0, "Error");
    TT_FATAL(bshape[-1] % in1_tile_shape[1] == 0, "Error");

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];

    if (fuse_batch) {
        Mt = B * Mt;
        B = 1;
    }
    TT_FATAL(Kt % in0_block_w == 0, "Error");

    // This should allocate a DRAM buffer on the device
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    uint32_t num_blocks_y = (Mt - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (Nt - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    // TODO: Max used grid can actually exceed mcast receiver grid if in0 is sharded
    // TODO: Move these validates to op validate and properly check for this
    TT_FATAL(
        num_blocks_total <= num_cores,
        "Number of blocks exceeds number of cores: {} blocks > {} cores",
        num_blocks_total,
        num_cores);

    if (!gather_in0) {
        TT_FATAL(hop_cores.empty(), "Hop cores are not supported for any mode besides gather_in0.");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    if (gather_in0) {
        return reuse_mcast_1d_optimized_helpers::create_program_gather_in0(
            a,
            b,
            device,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode,
            packer_l1_acc,
            dst_full_sync_en,
            compute_with_storage_grid_size,
            B,
            Mt,
            Nt,
            Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            per_core_M,
            per_core_N,
            fused_activation,
            hop_cores,
            in0_buffer,
            in1_buffer,
            out_buffer,
            in0_tile,
            in1_tile,
            output_tile,
            in0_data_format,
            in1_data_format,
            output_data_format,
            untilize_out,
            global_cb,
            num_global_cb_receivers,
            sub_device_id);
    }

    if (mcast_in0) {
        return reuse_mcast_1d_optimized_helpers::create_program_mcast_in0(
            a,
            device,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode,
            packer_l1_acc,
            compute_with_storage_grid_size,
            B,
            Mt,
            Nt,
            Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            out_block_w,
            per_core_M,
            per_core_N,
            fused_activation,
            in0_buffer,
            in1_buffer,
            bias_buffer,
            out_buffer,
            in0_tile,
            in1_tile,
            bias.has_value() ? bias->get_tensor_spec().tile() : output_tile,
            output_tile,
            in0_data_format,
            in1_data_format,
            bias_data_format,
            output_data_format,
            a.memory_config().is_sharded(),
            b.memory_config().is_sharded(),
            bias.has_value() ? bias->memory_config().is_sharded() : false,
            output.memory_config().is_sharded(),
            untilize_out,
            fused_op_signaler);
    } else {
        return reuse_mcast_1d_optimized_helpers::create_program_mcast_in1(
            device,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode,
            packer_l1_acc,
            compute_with_storage_grid_size,
            B,
            Mt,
            Nt,
            Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            out_block_w,
            per_core_M,
            per_core_N,
            fused_activation,
            in0_buffer,
            in1_buffer,
            bias_buffer,
            out_buffer,
            in0_tile,
            in1_tile,
            bias.has_value() ? bias->get_tensor_spec().tile() : output_tile,
            output_tile,
            in0_data_format,
            in1_data_format,
            bias_data_format,
            output_data_format,
            a.memory_config().is_sharded(),
            output.memory_config().is_sharded(),
            untilize_out);
    }
}

tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_mcast_1d_optimized(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool broadcast_batch,
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
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> empty_fused_op_signaler;

    return matmul_multi_core_reuse_mcast_1d_optimized_(
        a,
        b,
        bias,
        output_tensor,
        broadcast_batch,
        compute_with_storage_grid_size,
        compute_kernel_config,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        out_block_h,
        out_block_w,
        per_core_M,
        per_core_N,
        fuse_batch,
        std::move(fused_activation),
        mcast_in0,
        gather_in0,
        std::move(hop_cores),
        untilize_out,
        empty_fused_op_signaler,
        global_cb,
        num_global_cb_receivers,
        sub_device_id);
}

tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_mcast_1d_optimized_helper(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    MatmulMultiCoreReuseMultiCast1DProgramConfig config =
        std::get<MatmulMultiCoreReuseMultiCast1DProgramConfig>(program_config);

    return matmul_multi_core_reuse_mcast_1d_optimized_(
        a,
        b,
        bias,
        output_tensor,
        broadcast_batch,
        config.compute_with_storage_grid_size,
        compute_kernel_config,
        config.in0_block_w,
        config.out_subblock_h,
        config.out_subblock_w,
        config.out_block_h,
        config.out_block_w,
        config.per_core_M,
        config.per_core_N,
        config.fuse_batch,
        config.fused_activation,
        config.mcast_in0,
        config.gather_in0,
        config.hop_cores,
        untilize_out,
        fused_op_signaler,
        global_cb,
        config.num_global_cb_receivers,
        sub_device_id);
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn
