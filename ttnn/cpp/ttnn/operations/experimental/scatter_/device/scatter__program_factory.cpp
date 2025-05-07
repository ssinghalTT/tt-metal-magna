// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter__program_factory.hpp"

#include "scatter__device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::experimental::scatter {

Scatter_ProgramFactory::cached_program_t Scatter_ProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    Program program{};

    const auto& input_tensor{tensor_args.input_tensor};
    const auto& input_shape{input_tensor.get_padded_shape()};
    const auto& input_rank{input_shape.rank()};
    const auto& index_tensor{tensor_args.index_tensor};
    const auto& index_shape{index_tensor.get_padded_shape()};
    const auto& index_rank{index_shape.rank()};

    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    const tt::DataFormat index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(index_tensor.get_dtype());
    const tt::DataFormat output_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);
    const uint32_t output_tensor_tile_size = tile_size(output_tensor_cb_data_format);

    auto input_buffer = input_tensor.buffer();
    auto index_buffer = index_tensor.buffer();
    auto output_buffer = output_tensor.buffer();

    const bool input_tensor_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    const bool index_tensor_is_dram = index_buffer->buffer_type() == BufferType::DRAM;
    const bool output_tensor_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    const uint32_t num_input_tiles = input_tensor.volume() / TILE_HW;
    const uint32_t num_index_tiles = index_tensor.volume() / TILE_HW;
    const uint32_t num_output_tiles = output_tensor.volume() / TILE_HW;

    // const uint32_t input_volume = ;
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;
    const uint32_t Wt = input_shape[3] / TILE_WIDTH;

    // Double buffering config
    constexpr uint32_t num_cb_unit = 2;                // Number of circular buffer units for double buffering
    constexpr uint32_t cb_in_units = 2 * num_cb_unit;  // Total number of circular buffer units

    // Calculate the number of cores available for computation
    auto device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // // Calculate the number of cores utilized based on the input tensor shape
    // const uint32_t all_core_utilization_loop_count = Ht / total_number_of_cores;
    // const uint32_t all_core_utilization_loop_remainder = Ht % total_number_of_cores;

    // Calculate core range

    const CoreCoord core{0, 0};

    const int32_t dim{(args.dim >= 0) ? args.dim : (input_rank + args.dim)};

    //
    // const auto
    //     [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
    //         tt::tt_metal::split_work_to_cores(grid, num_rows_total);

    constexpr uint32_t input_tiles = 2;
    constexpr uint32_t index_tiles = 2;
    constexpr uint32_t out_tiles = 2;

    auto cb_input{create_cb(program, input_tensor.get_dtype(), ScatterCB::INPUT, CoreRangeSet{core}, input_tiles)};
    auto cb_index{create_cb(program, input_tensor.get_dtype(), ScatterCB::INDEX, CoreRangeSet{core}, index_tiles)};
    auto cb_dst{create_cb(program, input_tensor.get_dtype(), ScatterCB::DST, CoreRangeSet{core}, out_tiles)};

    constexpr std::array<const char*, 2> kernel_paths = {
        "ttnn/cpp/ttnn/operations/experimental/scatter/device/kernels/dataflow/reader_scatter_.cpp",
        //"ttnn/cpp/ttnn/operations/experimental/scatter/device/kernels/compute/compute.cpp",
        "ttnn/cpp/ttnn/operations/experimental/scatter/device/kernels/dataflow/writer_scatter_.cpp",
    };

    const std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_is_dram,
        index_tensor_is_dram,
        cb_input,
        cb_index,
        cb_dst,
    };

    const std::vector<uint32_t> writer_compile_time_args = {output_tensor_is_dram, cb_input, cb_index, cb_dst};

    auto reader_kernel = create_kernel(
        program, kernel_paths[0], CoreRangeSet{core}, ReaderDataMovementConfig{}, reader_compile_time_args);
    auto writer_kernel = create_kernel(
        program, kernel_paths[2], CoreRangeSet{core}, WriterDataMovementConfig{}, writer_compile_time_args);

    //

    return {std::move(program), {reader_kernel, writer_kernel, compute_with_storage_grid_size}};
}

void Scatter_ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    //
}

CBHandle Scatter_ProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const ScatterCB& cumprod_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& num_tiles) {
    using tt::tt_metal::detail::TileSize;
    const uint32_t cb_id{static_cast<uint32_t>(cumprod_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{TileSize(cb_data_format)};
    const auto cb_config{CircularBufferConfig{num_tiles * single_tile_size, {{cb_id, cb_data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle CumprodDeviceOperation::MultiCoreCumprodProgramFactory::create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);

    return kernel_id;
}

}  // namespace ttnn::operations::experimental::scatter
