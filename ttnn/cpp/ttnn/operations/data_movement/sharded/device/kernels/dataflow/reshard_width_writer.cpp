// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t shard_cb_id = get_compile_time_arg_val(0);

    // Kernel arguments
    uint32_t dst_addr = get_arg_val<uint32_t>(0);       // Starting destination address
    uint32_t read_offset = get_arg_val<uint32_t>(1);    // Offset in the circular buffer
    uint32_t num_writes = get_arg_val<uint32_t>(2);     // Number of write operations
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(3)); // Write parameters
    uint32_t args_idx = 0;

    // Starting address in the circular buffer
    uint32_t l1_read_addr = get_read_ptr(shard_cb_id) + read_offset;

    // Perform asynchronous writes for each operation
    for (uint32_t i = 0; i < num_writes; ++i) {
        uint32_t x_coord = args[args_idx++];            // Core X coordinate
        uint32_t y_coord = args[args_idx++];            // Core Y coordinate
        uint32_t addr = dst_addr + args[args_idx++];    // Destination address offset
        uint64_t dst_noc_addr = get_noc_addr(x_coord, y_coord, addr); // Construct NoC address
        uint32_t write_size = args[args_idx++];         // Size of the write
        noc_async_write(l1_read_addr, dst_noc_addr, write_size); // Perform the write
        l1_read_addr += write_size;                    // Advance circular buffer pointer
    }

    // Ensure all writes are complete
    noc_async_write_barrier();
}
