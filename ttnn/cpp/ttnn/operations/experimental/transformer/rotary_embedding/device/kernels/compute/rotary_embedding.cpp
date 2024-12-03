// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

#include "chlkc_pack_data_format.h"
#include "chlkc_unpack_data_format.h"
#include "debug/dprint.h"

#define DEBUG_COMPUTE 1

#ifdef DEBUG_COMPUTE
inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint8_t r = 0; r < 32; ++ r) {
        if (r % 8 == 0) {
            DPRINT << ENDL();
        }
        SliceRange sr = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ":  " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

inline void print_cb_details(uint32_t cb_id) {
    UNPACK(
        DPRINT << "cb_id " << cb_id << ": { "
               << "unpack data format: " << (uint)unpack_src_format[cb_id] << ", "
               << "size: " << cb_interface[cb_id].fifo_size << ", "
               << "limit: " << cb_interface[cb_id].fifo_limit << ", "
               << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
               << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
               << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
               << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
               << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL());
}

inline void print_two_tiles(uint32_t cb_id) {
    cb_wait_front(cb_id, 2);
    print_cb_details(cb_id);
    tensix_sync();
    UNPACK(print_full_tile(cb_id, 0, false));
    UNPACK(print_full_tile(cb_id, 1, false));
    tensix_sync();
}
#endif

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

ALWI void MUL_TILES(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles, uint32_t in1_idx) {
    // Multiply input by cos
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, in1_idx + 1);
    cb_reserve_back(out_cb, num_tiles);

    ACQ();
    mul_bcast_rows_init_short();
    mul_tiles_bcast_rows(in0_cb, in1_cb, 0, in1_idx, 0);
    pack_tile(0, out_cb);
    REL();
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
}

ALWI void UNTILIZE_TILES(uint32_t in0_cb, uint32_t out_cb, uint32_t num_tiles) {
    untilize_init_short(in0_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    untilize_block(in0_cb, num_tiles, out_cb);
    cb_push_back(out_cb, num_tiles);

#ifdef DEBUG_COMPUTE
    UNPACK(DPRINT << "Tilized original sin tiles " << ENDL(););
    print_two_tiles(in0_cb);
#endif

    cb_pop_front(in0_cb, num_tiles);
    untilize_uninit(in0_cb);
}

ALWI void TILIZE_ROWS(uint32_t in0_cb, uint32_t sync_cb, uint32_t out_cb, uint32_t num_tiles) {
    tilize_init_short(in0_cb, num_tiles);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(sync_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    tilize_block(in0_cb, num_tiles, out_cb);
    cb_push_back(out_cb, num_tiles);

    // Pop shared cbs after tilize
    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(sync_cb, num_tiles);
    tilize_uninit(in0_cb);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(3);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    uint32_t updated_sin_cb = sin_cb;

    constexpr uint32_t untilized_sin_cb = get_compile_time_arg_val(14);
    constexpr uint32_t untilized_sin_sync_cb = get_compile_time_arg_val(15);
    constexpr uint32_t retilized_sin_cb = get_compile_time_arg_val(17);

    binary_op_init_common(sin_cb, untilized_sin_sync_cb, untilized_sin_cb);
    UNTILIZE_TILES(sin_cb, untilized_sin_cb, Wt);

#ifdef DEBUG_COMPUTE
    UNPACK(DPRINT << "Untilized sin tiles " << ENDL(););
    print_two_tiles(untilized_sin_cb);
#endif

    reconfig_data_format_srca(sin_cb, untilized_sin_cb);
    pack_reconfig_data_format(untilized_sin_cb, retilized_sin_cb);
    TILIZE_ROWS(untilized_sin_cb, untilized_sin_sync_cb, retilized_sin_cb, Wt);
    updated_sin_cb = retilized_sin_cb;

#ifdef DEBUG_COMPUTE
    UNPACK(DPRINT << "Retilized sin tiles " << ENDL(););
    print_two_tiles(retilized_sin_cb);
#endif

    /*
        uint32_t in1_idx = 0;
        for (uint32_t i = 0; i < num_rows; ++i) {
            for (uint32_t j = 0; j < Wt; ++j) {
                in1_idx = j;
                reconfig_data_format(in_cb, updated_sin_cb);
                pack_reconfig_data_format(retilized_sin_cb, out_cb);
                MUL_TILES(in_cb, updated_sin_cb, out_cb, onetile, in1_idx);
            }
        }
    */
}
}  // namespace NAMESPACE
