// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("UNPACK-BFP8-TILE");
    bool verify_mode = get_arg_val<uint32_t>(3) == 0 ? true : false;
    if (!verify_mode)
        return;

    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t out_dram_noc_x = get_arg_val<uint32_t>(1);
    uint32_t out_dram_noc_y = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
    uint64_t out_noc_addr = get_noc_addr(out_dram_noc_x, out_dram_noc_y, out_addr);
    cb_wait_front(cb_id_out0, 1);
    noc_async_write(l1_read_addr, out_noc_addr, get_tile_size(cb_id_out0));
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, 1);
}
