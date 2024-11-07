// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src0_num_tiles  = get_arg_val<uint32_t>(1);
    uint32_t src1_addr  = get_arg_val<uint32_t>(2);
    uint32_t src1_num_tiles  = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);
    uint32_t ublock_size_tiles = 1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_tiles = src0_num_tiles > src1_num_tiles ? src0_num_tiles : src1_num_tiles;
    InterleavedAddrGen<true> src_0_addr_gen{.bank_base_address = src0_addr, .page_size = ublock_size_bytes_0 * num_tiles};
    InterleavedAddrGen<true> src_1_addr_gen{.bank_base_address = src1_addr, .page_size = ublock_size_bytes_1 * num_tiles};
    uint32_t src0_addr_offset = 0;
    uint32_t src1_addr_offset = 0;
    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i=0; i<num_tiles; i += ublock_size_tiles) {
        if (i < src0_num_tiles) {
            uint64_t src0_noc_addr = src_0_addr_gen.get_noc_addr(0, src0_addr_offset);

            cb_reserve_back(cb_id_in0, ublock_size_tiles);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);

            noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);

            noc_async_read_barrier();

            cb_push_back(cb_id_in0, ublock_size_tiles);

            src0_addr_offset += ublock_size_bytes_0;
        }

        if (i < src1_num_tiles) {
            uint64_t src1_noc_addr = src_1_addr_gen.get_noc_addr(0, src1_addr_offset);

            cb_reserve_back(cb_id_in1, ublock_size_tiles);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);

            noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);

            noc_async_read_barrier();

            cb_push_back(cb_id_in1, ublock_size_tiles);

            src1_addr_offset += ublock_size_bytes_1;
        }
    }
}
