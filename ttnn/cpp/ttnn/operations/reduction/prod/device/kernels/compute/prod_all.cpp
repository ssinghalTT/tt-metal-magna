// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "tt_metal/include/compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/negative.h"

#include "tt_metal/hw/inc/debug/dprint_pages.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
    bool last_tile = false;
    bool once = true;
    cb_reserve_back(tt::CBIndex::c_3, 1);
    for (uint32_t t = 0; t < num_tiles; t++) {
        if (t == (num_tiles - 1)) {
            last_tile = true;
        }

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(tt::CBIndex::c_0, 1);
            // PACK(tt::compute::common::print_full_tile(tt::CBIndex::c_0));
            if (once) {
                cb_reserve_back(tt::CBIndex::c_2, 1);
                tile_regs_acquire();
                copy_tile_to_dst_init_short(tt::CBIndex::c_0);
                copy_tile(tt::CBIndex::c_0, 0, 0);  // copy from c_in[0] to DST[0]
                tile_regs_commit();
                tile_regs_wait();
                if constexpr (num_tiles == 1) {
                    pack_tile(0, tt::CBIndex::c_3);
                } else {
                    pack_tile(0, tt::CBIndex::c_2);
                    cb_push_back(tt::CBIndex::c_2, 1);
                }
                tile_regs_release();
            } else {
                tile_regs_acquire();
                mul_tiles_init(tt::CBIndex::c_0, tt::CBIndex::c_2);
                mul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                if (last_tile) {
                    pack_tile(0, tt::CBIndex::c_3);
                } else {
                    cb_pop_front(tt::CBIndex::c_2, 1);
                    cb_reserve_back(tt::CBIndex::c_2, 1);
                    pack_tile(0, tt::CBIndex::c_2);
                    cb_push_back(tt::CBIndex::c_2, 1);
                }
                tile_regs_release();
            }
            once = false;
            cb_pop_front(tt::CBIndex::c_0, 1);
        }
    }
    cb_push_back(tt::CBIndex::c_3, 1);
}
}  // namespace NAMESPACE
