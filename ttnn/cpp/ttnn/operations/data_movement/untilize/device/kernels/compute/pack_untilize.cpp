// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    DPRINT << "per_core_block_cnt: " << per_core_block_cnt << ENDL();
    DPRINT << "per_core_block_tile_cnt: " << per_core_block_tile_cnt << ENDL();

    pack_untilize_init<per_core_block_tile_cnt>(tt::CB::c_in0, tt::CB::c_out0);

    for(uint32_t b = 0; b < per_core_block_cnt; ++ b) {
        cb_wait_front(tt::CB::c_in0, per_core_block_tile_cnt);
        if (b == 1) {
            DPRINT_UNPACK({ DPRINT << "Second tile first row before untilizing: " << ENDL(); });
            DPRINT_UNPACK({ DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::h0_w0_32()) << ENDL(); });
        }
        cb_reserve_back(tt::CB::c_out0, per_core_block_tile_cnt);

        pack_untilize_block<per_core_block_tile_cnt>(tt::CB::c_in0, 1, tt::CB::c_out0);
        if (b == 1) {
            DPRINT_PACK({ DPRINT << "Second tile first row after untilizing: " << ENDL(); });
            auto rd_ptr = cb_interface[tt::CB::c_out0].fifo_wr_ptr << 4;
            volatile tt_l1_ptr uint16_t* address_map = (volatile tt_l1_ptr uint16_t*)(rd_ptr);
            for (uint32_t i = 0; i < 32; ++i) {
                DPRINT_PACK({ DPRINT << BF16(address_map[i]) << " "; });
            }
            DPRINT_PACK({ DPRINT << ENDL(); });
        }

        cb_push_back(tt::CB::c_out0, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::c_in0, per_core_block_tile_cnt);
    }

    pack_untilize_uninit();
}
}
