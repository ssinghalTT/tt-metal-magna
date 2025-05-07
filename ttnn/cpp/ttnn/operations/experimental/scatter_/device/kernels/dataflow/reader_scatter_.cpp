// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../scatter__common.hpp"

void kernel_main() {
    // Runtime args
    const uint32_t input_index_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    // const uint32_t core_loop_count = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr bool input_index_tensor_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t input_tensor_cb = get_compile_time_arg_val(1);
    constexpr uint32_t index_tensor_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_tensor_cb = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(5);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(6);
    // constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(7);
    // constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(8);
    // constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(9);

    // Input tensor config
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_cb);
    constexpr DataFormat input_tensor_data_format = get_dataformat(input_tensor_cb);
    const InterleavedAddrGenFast<input_tensor_is_dram> input_tensor_dram = {
        .bank_base_address = input_tensor_buffer_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Index tensor config
    constexpr uint32_t index_tensor_tile_size_bytes = get_tile_size(index_tensor_cb);
    constexpr DataFormat index_tensor_data_format = get_dataformat(index_tensor_cb);
    const InterleavedAddrGenFast<index_tensor_is_dram> index_tensor_dram = {
        .bank_base_address = input_tensor_buffer_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Output tensor config
    constexpr uint32_t output_tensor_tile_size_bytes = get_tile_size(output_tensor_cb_index);
    constexpr DataFormat output_tensor_data_format = get_dataformat(output_tensor_cb_index);
    const InterleavedAddrGenFast<output_tensor_is_dram> output_tensor_dram = {
        .bank_base_address = output_tensor_buffer_addr,
        .page_size = output_tensor_tile_size_bytes,
        .data_format = output_tensor_data_format};

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Read input data
        for (uint32_t w = 0; w < Wt_input; w++) {
            cb_reserve_back(input_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_index);
            noc_async_read_tile(h * Wt_input + w, input_tensor_dram, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(input_tensor_cb_index, one_tile);
        }

        // Write output data
        for (uint32_t w = 0; w < Wt_index; w++) {
            cb_wait_front(output_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_output = get_read_ptr(output_tensor_cb_index);
            noc_async_write_tile(h * Wt_index + w, output_tensor_dram, l1_write_addr_output);
            noc_async_write_barrier();
            cb_pop_front(output_tensor_cb_index, one_tile);
        }
    }
}
