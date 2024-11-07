// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);
    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(2);
    std::uint32_t read_size             = get_arg_val<uint32_t>(3);
    std::uint32_t dram_buffer_page_size = get_arg_val<uint32_t>(4);

    InterleavedAddrGen<true> src_addr_gen{.bank_base_address = dram_buffer_src_addr, .page_size = dram_buffer_page_size};
    InterleavedAddrGen<true> dst_addr_gen{.bank_base_address = dram_buffer_dst_addr, .page_size = dram_buffer_page_size};

    std::uint64_t dram_buffer_src_noc_addr = src_addr_gen.get_noc_addr(0, 0);
    noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, read_size);
    noc_async_read_barrier();

    std::uint64_t dram_buffer_dst_noc_addr = dst_addr_gen.get_noc_addr(0, 0);
    noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, read_size);
    noc_async_write_barrier();
}
