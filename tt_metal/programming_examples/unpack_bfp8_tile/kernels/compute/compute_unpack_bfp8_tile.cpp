// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    DeviceZoneScopedN("UNPACK-BFP8-TILE");
    bool unpack_to_bf16 = get_arg_val<uint32_t>(0) > 0 ? true : false;
    if (unpack_to_bf16)
        return;

    auto in0_cb_id = tt::CB::c_in0;
    auto in1_cb_id = tt::CB::c_in1;
    auto out_cb_id = tt::CB::c_out0;

    binary_op_init_common(in0_cb_id, in1_cb_id, out_cb_id);
    add_tiles_init();

    cb_wait_front(in0_cb_id, 1);
    tile_regs_acquire();  // acquire 8 tile registers
    add_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);
    tile_regs_commit();  // signal the packer
    cb_pop_front(in0_cb_id, 1);

    cb_reserve_back(out_cb_id, 1);
    tile_regs_wait();  // packer waits here
    pack_tile(0, out_cb_id);
    tile_regs_release();  // packer releases
    cb_push_back(out_cb_id, 1);
}
}  // namespace NAMESPACE
