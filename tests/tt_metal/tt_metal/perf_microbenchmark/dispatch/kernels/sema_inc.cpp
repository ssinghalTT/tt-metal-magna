// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "tt_metal/hw/inc/dataflow_api.h"
#include "tt_metal/hw/inc/risc_attribs.h"


void kernel_main() {
    // required macros: START, OTHER_ADDR_X, OTHER_ADDR_Y, ITERATIONS
    auto semaphore_id = get_arg_val<uint32_t>(0);

    // get a raw volatile ptr to the semaphore
    //  ➩ must be volatile otherwise compiler will assume it cannot change!
    uint32_t sema_addr = get_semaphore(semaphore_id);
    volatile tt_l1_ptr uint32_t* sema_ptr = (tt_l1_ptr uint32_t*)sema_addr;

    // set semaphore initial value according to comptime MACRO; only one kernel should start at 1!
    *sema_ptr = START;

    // pull noc address calculation outside of loop
    std::uint64_t noc_addr = get_noc_addr(OTHER_ADDR_X, OTHER_ADDR_Y, sema_addr);

    uint32_t prev = 0;
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        //-- wait for signal from other core
        while (*sema_ptr == prev);

        // update prev value to detect next increment
        prev = *sema_ptr;

        //-- update semaphore in the other core!
        noc_fast_atomic_increment<noc_mode>(
            noc_index,
            write_at_cmd_buf,
            noc_addr,
            NOC_UNICAST_WRITE_VC,
            1 /*increment*/,
            31 /*wrap*/,
            false /*linked*/,
            true /*posted*/,
            MEM_NOC_ATOMIC_RET_VAL_ADDR);
    }
}