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
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

// clang-format on

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t target_address = get_compile_time_arg_val(2);
constexpr bool mcast_mode = get_compile_time_arg_val(3);

inline void setup_connection_and_headers(
    tt::tt_fabric::WorkerToFabricEdmSender& connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t hops,
    uint64_t noc_dest_addr,
    uint32_t packet_payload_size_bytes) {
    // connect to edm
    connection.open();

    if constexpr (mcast_mode) {
        packet_header->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(hops)});
    } else {
        packet_header->to_chip_unicast(static_cast<uint8_t>(hops));
    }

    packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_addr}, packet_payload_size_bytes);
}

inline void send_packet(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint64_t noc_dest_addr,
    uint32_t source_l1_buffer_address,
    uint32_t packet_payload_size_bytes,
    uint32_t seed,
    tt::tt_fabric::WorkerToFabricEdmSender& connection) {
#ifndef BENCHMARK_MODE
    packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_addr}, packet_payload_size_bytes);
    // fill packet data for sanity testing
    tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
    fill_packet_data(start_addr, packet_payload_size_bytes / 16, seed);
    tt_l1_ptr uint32_t* last_word_addr =
        reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address + packet_payload_size_bytes - 4);
#endif
    connection.wait_for_empty_write_slot();
    connection.send_payload_without_header_non_blocking_from_address(
        source_l1_buffer_address, packet_payload_size_bytes);
    connection.send_payload_blocking_from_address((uint32_t)packet_header, sizeof(tt::tt_fabric::PacketHeader));
}

void kernel_main() {
    using namespace tt::tt_fabric;

    size_t rt_args_idx = 0;
    uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t rx_noc_encoding = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_args_idx++);

    uint64_t noc_dest_addr = get_noc_addr_helper(rx_noc_encoding, 0x80000);

    auto fwd_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    packet_header->to_chip_unicast(1);
    packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc_dest_addr}, 4);

    auto data_pointer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(0x80000 + sizeof(PACKET_HEADER_TYPE));

    for (uint32_t i = 0; i < 64; ++i) {
        data_pointer[i] = 0xdeadbeef;
    }

    fwd_fabric_connection.wait_for_empty_write_slot();
    // Set data
    fwd_fabric_connection.send_payload_without_header_non_blocking_from_address(
        (uint32_t)data_pointer, sizeof(uint32_t) * 64);
    // Set header
    fwd_fabric_connection.send_payload_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));

    noc_async_write_barrier();
}
