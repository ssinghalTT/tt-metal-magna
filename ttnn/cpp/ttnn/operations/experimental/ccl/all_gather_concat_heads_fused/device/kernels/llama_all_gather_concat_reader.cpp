// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);
constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(3);
constexpr uint32_t SUBTILE_LINE_BYTES = get_compile_time_arg_val(4);
constexpr uint32_t cb_id_q_out = get_compile_time_arg_val(5);
constexpr uint32_t head_size = get_compile_time_arg_val(6);
constexpr uint32_t batch = get_compile_time_arg_val(7);
constexpr uint32_t head_size_num_tiles = get_compile_time_arg_val(8);
constexpr uint32_t PHASES_TO_READ =
    get_compile_time_arg_val(9);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase

constexpr uint32_t in_num_cores = get_compile_time_arg_val(10);
constexpr uint32_t face_h = get_compile_time_arg_val(11);
constexpr uint32_t face_hw = get_compile_time_arg_val(12);

constexpr uint32_t temp_cb_id = get_compile_time_arg_val(13);
constexpr uint32_t batch_size = get_compile_time_arg_val(14);
constexpr uint32_t batch_start_1 = get_compile_time_arg_val(15);
constexpr uint32_t batch_end_1 = get_compile_time_arg_val(16);
constexpr uint32_t batch_start_2 = get_compile_time_arg_val(17);
constexpr uint32_t batch_end_2 = get_compile_time_arg_val(18);
constexpr uint32_t start_local = get_compile_time_arg_val(19);
constexpr uint32_t tile_size = get_compile_time_arg_val(20);
constexpr uint32_t num_tiles_per_core_concat = get_compile_time_arg_val(21);

void batch_loop(
    uint32_t q_start_addr,
    uint32_t tensor_address0,
    const uint32_t cb_write_ptr_base,
    uint64_t qkv_read_addr,
    uint32_t num_tiles_read_cur_core,
    uint32_t cur_core_idx,
    uint32_t start,
    uint32_t end,
    uint32_t concat_arg_start,
    uint32_t local_count,
    tt_l1_ptr uint32_t* core_noc_x,
    tt_l1_ptr uint32_t* core_noc_y,
    tt_l1_ptr uint32_t* in0_mcast_noc_x,
    tt_l1_ptr uint32_t* in0_mcast_noc_y,
    bool nlp_local,
    uint32_t phase,
    uint32_t in_tile_offset_by_head) {
    for (uint32_t q = start; q < end; ++q) {
        uint32_t wptr_offset = q < face_h ? q * SUBTILE_LINE_BYTES : (q + face_h) * SUBTILE_LINE_BYTES;
        uint32_t q_write_addr = cb_write_ptr_base + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            if (phase == 0 || phase == 1) {
                noc_async_read(qkv_read_addr, q_write_addr, SUBTILE_LINE_BYTES);
            }
            // Read second phase
            if (phase == 0 || phase == 2) {
                noc_async_read(
                    qkv_read_addr + face_hw * ELEMENT_SIZE, q_write_addr + face_hw * ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            }
            qkv_read_addr += tile_size;
            q_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core_concat) {
                cur_core_idx++;
                local_count++;
                qkv_read_addr =
                    get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                    in_tile_offset_by_head;
                if (nlp_local) {
                    qkv_read_addr = get_noc_addr(core_noc_x[local_count], core_noc_y[local_count], tensor_address0) +
                                    in_tile_offset_by_head;
                }
                num_tiles_read_cur_core = 0;
            }
        }
    }
};

void nlp_concat(
    uint32_t batch,
    uint32_t q_start_addr,
    uint32_t tensor_address0,
    uint32_t head_size,
    uint32_t cb_id_q_out,
    uint32_t phase,
    uint32_t concat_arg_start,
    bool nlp_local,
    uint32_t start_local,
    tt_l1_ptr uint32_t* core_noc_x,
    tt_l1_ptr uint32_t* core_noc_y,
    tt_l1_ptr uint32_t* in0_mcast_noc_x,
    tt_l1_ptr uint32_t* in0_mcast_noc_y,
    uint32_t in_tile_offset_by_head) {
    // Q
    uint32_t cur_core_idx = batch_start_1;

    uint64_t qkv_read_addr = get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                             in_tile_offset_by_head;
    uint32_t local_count = 0;
    if (nlp_local) {
        qkv_read_addr =
            get_noc_addr(core_noc_x[local_count], core_noc_y[local_count], tensor_address0) + in_tile_offset_by_head;
    }

    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    const uint32_t cb_write_ptr_base = get_write_ptr(cb_id_q_out);

    uint32_t start = nlp_local ? start_local : batch_start_1;
    uint32_t end = nlp_local ? start_local + 8 : batch_end_1;
    uint32_t idx_end = nlp_local ? 1 : batch_size;
    for (uint32_t batch_range = 0; batch_range < idx_end; batch_range++) {
        batch_loop(
            q_start_addr,
            tensor_address0,
            cb_write_ptr_base,
            qkv_read_addr,
            num_tiles_read_cur_core,
            cur_core_idx,
            start,
            end,
            concat_arg_start,
            local_count,
            core_noc_x,
            core_noc_y,
            in0_mcast_noc_x,
            in0_mcast_noc_y,
            nlp_local,
            phase,
            in_tile_offset_by_head);
        start = batch_start_2;
        end = batch_end_2;
        cur_core_idx = batch_start_2;
        qkv_read_addr = get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                        in_tile_offset_by_head;

        num_tiles_read_cur_core = 0;
    }
    noc_async_read_barrier();
};

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 1;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t out_ready_sem_bank_addr_concat = get_arg_val<uint32_t>(arg_idx++);
    constexpr uint32_t num_tiles_per_core = 4;  // get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    uint32_t concat_arg_start = get_arg_val<uint32_t>(0);
    uint32_t in_tile_offset_by_head = get_arg_val<uint32_t>(concat_arg_start);
    uint32_t q_start_addr = get_arg_val<uint32_t>(concat_arg_start + 1);

    uint32_t arg_sem_idx = 2 + 2 * in_num_cores;
    constexpr uint32_t out_ready_sem_wait_value_concat = 12;  // get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx);
    constexpr uint32_t out_ready_sem_noc0_x_concat = 19;  // get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx + 1);
    constexpr uint32_t out_ready_sem_noc0_y_concat = 18;  // get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx + 2);
    uint32_t is_drain_core = get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx);
    const uint32_t signal_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx + 1));

    uint32_t local_arg = concat_arg_start + arg_sem_idx + 2;
    tt_l1_ptr uint32_t* nlp_local_core_x = (tt_l1_ptr uint32_t*)(get_arg_addr(local_arg));
    local_arg += 8;
    tt_l1_ptr uint32_t* nlp_local_core_y = (tt_l1_ptr uint32_t*)(get_arg_addr(local_arg));
    local_arg += 8;
    tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + concat_arg_start));
    tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + in_num_cores + concat_arg_start));

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        cb_reserve_back(cb0_id, num_tiles_to_read_this_core);

        const uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t read_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0);
        read_addr += shard_tile_id * tensor0_page_size;

        noc_async_read(read_addr, l1_write_addr, num_tiles_to_read_this_core * tensor0_page_size);
        noc_async_read_barrier();

        cb_push_back(cb0_id, num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }

    nlp_concat(
        batch,
        q_start_addr,
        tensor_address0,
        head_size,
        cb_id_q_out,
        0,
        concat_arg_start,
        1,
        start_local,
        nlp_local_core_x,
        nlp_local_core_y,
        in0_mcast_noc_x,
        in0_mcast_noc_y,
        in_tile_offset_by_head);

    uint64_t out_ready_sem_noc_addr_concat =
        safe_get_noc_addr(out_ready_sem_noc0_x_concat, out_ready_sem_noc0_y_concat, out_ready_sem_bank_addr_concat);

    noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);

    uint32_t phase = PHASES_TO_READ;
    nlp_concat(
        batch,
        q_start_addr,
        tensor_address0,
        head_size,
        cb_id_q_out,
        phase,
        concat_arg_start,
        0,
        start_local,
        nlp_local_core_x,
        nlp_local_core_y,
        in0_mcast_noc_x,
        in0_mcast_noc_y,
        in_tile_offset_by_head);
}
