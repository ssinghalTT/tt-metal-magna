// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>
#include "dataflow_api.h"

#define ENABLE_DEBUG_PRINT 1

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

#define ALWI inline __attribute__((always_inline))

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n / 2; ++i) {
        ptr[i] = (val | (val << 16));
    }
    return true;
}

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA0({ DPRINT << r << " " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL(); });
    }
    DPRINT << "++++++" << ENDL();
}

/**
 * Pool 2D (Max pool 2D and Avg pool 2D)
 */
void kernel_main() {
    // compile time args
    constexpr uint32_t reader_nindices = get_compile_time_arg_val(0);
    constexpr uint32_t window_h = get_compile_time_arg_val(1);
    constexpr uint32_t window_w = get_compile_time_arg_val(2);

    constexpr int32_t pad_w = get_compile_time_arg_val(3);

    // channel size in bytes
    constexpr uint32_t in_nbytes_c = get_compile_time_arg_val(4);

    // input tensor height / width / channels
    constexpr int32_t in_w = get_compile_time_arg_val(5);

    constexpr uint32_t in_c = get_compile_time_arg_val(6);

    constexpr uint32_t split_reader = get_compile_time_arg_val(7);
    constexpr uint32_t reader_id = get_compile_time_arg_val(8);
    constexpr uint32_t ceil_pad_w = get_compile_time_arg_val(14);

    constexpr uint32_t TILE_WIDTH = 32;

    constexpr uint32_t in_cb_id = (reader_id == 1) ? get_compile_time_arg_val(16) : get_compile_time_arg_val(15);
    constexpr uint32_t in_shard_cb_id = get_compile_time_arg_val(17);
    constexpr uint32_t in_reader_indices_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in_scalar_cb_id = get_compile_time_arg_val(19);

    const uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint16_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reader_indices_l1_addr);

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;

    constexpr uint32_t npages_to_reserve = 1;
    uint32_t counter = reader_id;
    uint32_t scalar_index = 0;
    uint32_t scalars_cnt = get_arg_val<uint32_t>(0);
    DPRINT << "scalars_cnt: " << scalars_cnt << ENDL();
    while (counter < reader_nindices || (reader_id == 0 && scalar_index < scalars_cnt)) {
        if (reader_id == 0 && scalar_index < scalars_cnt) {
            uint32_t scalar_val = get_arg_val<uint32_t>(scalar_index + 1);
            DPRINT << scalar_index << ENDL();
            cb_reserve_back(in_scalar_cb_id, 1);
            fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_WIDTH, scalar_val >> 16);
            scalar_index++;
            cb_push_back(in_scalar_cb_id, 1);
            // print_full_tile(in_scalar_cb_id, scalar_index - 1);
        }
        if (counter < reader_nindices) {
            cb_reserve_back(in_cb_id, npages_to_reserve);
            uint32_t out_l1_write_addr = get_write_ptr(in_cb_id);
            uint16_t top_left_local_index = reader_indices_ptr[counter++];
            DPRINT << "top left local index: " << top_left_local_index << ENDL();
            uint32_t h_multiples = 0;
            for (uint32_t h = 0; h < window_h; ++h, h_multiples += in_w_padded) {
                const uint32_t stick_offset = top_left_local_index + h_multiples;
                const uint32_t read_offset = in_l1_read_base_addr + (stick_offset * in_nbytes_c);
                noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, in_nbytes_c * window_w);
                out_l1_write_addr += in_nbytes_c * window_w;
            }
            if (split_reader) {
                counter++;  // interleave the indices
            }
            noc_async_read_barrier();
            cb_push_back(in_cb_id, npages_to_reserve);
        }
    }
    DPRINT << "kraj reada" << ENDL();
}  // kernel_main()
