// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    uint32_t cb_in = tt::CBIndex::c_0;
    uint32_t cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_in, cb_out);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_out, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(cb_in, 1);
            copy_tile(cb_in, 0, 0);

            typecast_tile_init();
            typecast_tile<(uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::UInt16>(0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);

            cb_pop_front(cb_in, 1);
            tile_regs_release();
        }
        PACK((pack_reconfig_data_format(cb_out)));
        cb_push_back(cb_out, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
