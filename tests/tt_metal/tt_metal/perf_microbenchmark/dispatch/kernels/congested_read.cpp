// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
    uint32_t src_x = get_arg_val<uint32_t>(0);
    uint32_t src_y = get_arg_val<uint32_t>(1);

    uint32_t cb_addr = get_write_ptr(0) + (8192*src_y);
    uint32_t read_ptr = cb_addr;

    uint64_t noc_addr = get_noc_addr(src_x, src_y, read_ptr);

    for (int i = 0; i < ITERATIONS; i++) {
        for (int j = 0; j < PAGE_COUNT; j++) {
            noc_async_read(noc_addr, read_ptr, PAGE_SIZE);
            noc_async_read_barrier();
        }
    }
}
