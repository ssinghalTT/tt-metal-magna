// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("UNPACK-BFP8-TILE");
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_dram_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src_dram_noc_y = get_arg_val<uint32_t>(2);
    bool unpack_to_bf16 = get_arg_val<uint32_t>(3) > 0 ? true : false;
    bool verify_mode = get_arg_val<uint32_t>(4) == 0 ? true : false;

    uint64_t src_noc_addr = get_noc_addr(src_dram_noc_x, src_dram_noc_y, src_addr);

    constexpr uint32_t in0_cb_id = tt::CB::c_in0;
    cb_reserve_back(in0_cb_id, 1);
    uint32_t l1_write_addr_in = get_write_ptr(in0_cb_id);
    if (verify_mode) {
        noc_async_read(src_noc_addr, l1_write_addr_in, get_tile_size(in0_cb_id));
        noc_async_read_barrier();
    }
    cb_push_back(in0_cb_id, 1);

    if (unpack_to_bf16) {
        constexpr uint32_t out_cb_id = tt::CB::c_out0;
        cb_reserve_back(out_cb_id, 1);
        uint32_t l1_write_addr_out = get_write_ptr(out_cb_id);
        auto in_addr = reinterpret_cast<uint8_t *>(l1_write_addr_in);
        auto out_addr = reinterpret_cast<uint16_t *>(l1_write_addr_out);
        for (uint32_t shared_exp_byte_id = 0; shared_exp_byte_id < 64; ++shared_exp_byte_id) {
            uint8_t shared_exp = in_addr[shared_exp_byte_id];
            for (uint32_t sign_mantissa_byte_id = 0; sign_mantissa_byte_id < 16; ++sign_mantissa_byte_id) {
                uint8_t sign_mantissa = in_addr[sign_mantissa_byte_id + shared_exp_byte_id * 16 + 64];
                bool sign = sign_mantissa & (1 << 7);
                uint8_t mantissa = sign_mantissa & ~(1 << 7);  // set sign bit to zero
                uint8_t exp = shared_exp;

                uint16_t bf16 = 0;
                if (mantissa != 0) {
                    while ((mantissa & (1 << 6)) == 0) {
                        mantissa <<= 1;
                        exp--;
                    }
                    // Do another shift and clear the hidden bit
                    mantissa <<= 1;
                    mantissa &= ~(1 << 7);

                    bf16 |= mantissa;
                    bf16 |= (exp << 7);
                    bf16 |= (sign << 15);
                }
                *out_addr = bf16;
                out_addr++;
            }
        }
        cb_push_back(out_cb_id, 1);
    } else {
        constexpr uint32_t in1_cb_id = tt::CB::c_in1;
        cb_reserve_back(in1_cb_id, 1);
        auto in1_addr = reinterpret_cast<uint8_t *>(get_write_ptr(in1_cb_id));
        for (uint32_t i = 0; i < get_tile_size(in1_cb_id); i++) in1_addr[i] = 0;
        cb_push_back(in1_cb_id, 1);
    }
}
