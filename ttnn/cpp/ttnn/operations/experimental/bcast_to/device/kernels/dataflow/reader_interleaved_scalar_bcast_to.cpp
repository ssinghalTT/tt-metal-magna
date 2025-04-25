// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t HtWt = get_arg_val<uint32_t>(3);
    uint32_t n_stride = get_arg_val<uint32_t>(4);
    uint32_t c_stride = get_arg_val<uint32_t>(5);
    uint32_t N = get_arg_val<uint32_t>(6);
    uint32_t C = get_arg_val<uint32_t>(7);
    uint32_t Ht = get_arg_val<uint32_t>(8);
    uint32_t Wt = get_arg_val<uint32_t>(9);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr auto cb_id_src = tt::CBIndex::c_0;
    constexpr uint32_t onetile = 1;

    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const DataFormat src_data_format = get_dataformat(cb_id_src);
    const InterleavedAddrGenFast<src_is_dram> src = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};
    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    // this is the INPUT tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride;
    uint32_t next_batch_shift = n_stride - c_stride * C;
    uint32_t next_channel_shift = c_stride - HtWt;

    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_t = 0) {
            cb_reserve_back(cb_id_src, onetile);
            uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);
            noc_async_read_tile(tile_offset, src, l1_write_addr_src);
            noc_async_read_barrier();
            fill_tile_with_first_element_bfloat16(cb_id_src);
            cb_push_back(cb_id_src, onetile);
            num_tiles_read += HtWt - start_t;
            tile_offset += c_stride;
            // same as following logically
            // tile_offset += HtWt;
            // tile_offset += next_channel_shift;
        }
        tile_offset += next_batch_shift;
    }
}
