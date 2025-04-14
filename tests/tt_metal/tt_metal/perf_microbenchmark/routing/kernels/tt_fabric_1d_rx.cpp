// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"

// clang-format on

void kernel_main() {
    DPRINT << "Waiting for data" << ENDL();
    auto data_pointer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(0x80000);
    while (data_pointer[0] == 0);
    DPRINT << "Data received 0x" << HEX() << *data_pointer << ENDL();
}
