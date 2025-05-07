// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_edm_packet_header.hpp"
#include "debug/assert.h"

namespace tt::tt_fabric {

FORCE_INLINE void validate(const PacketHeader& packet_header) {
    ASSERT(packet_header.chip_send_type <= CHIP_SEND_TYPE_LAST);
}
FORCE_INLINE bool is_valid(const PacketHeader& packet_header) {
    return (packet_header.chip_send_type <= CHIP_SEND_TYPE_LAST) && (packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

FORCE_INLINE void validate_detailed(volatile LowLatencyPacketHeader* packet_header) {
    ASSERT(packet_header->noc_send_type <= NOC_SEND_TYPE_LAST);
    ASSERT(packet_header->payload_size_bytes <= 4 * 1088);  // TODO: get max packet size from somewhere
    switch (packet_header->noc_send_type) {
        case NocSendType::NOC_UNICAST_WRITE:
            ASSERT(((packet_header->command_fields.unicast_write.noc_address >> 36) & 0x3F) <= 36);
            ASSERT(((packet_header->command_fields.unicast_write.noc_address >> 42) & 0x3F) <= 36);
            break;

        case NocSendType::NOC_MULTICAST_WRITE:
            // ASSERT(((packet_header.command_fields.mcast_write.noc_x_start >> 36) & 0x3F) <= 36);
            // ASSERT(((packet_header.command_fields.mcast_write.noc_y_start >> 42) & 0x3F) <= 36);
            // ASSERT(((packet_header.command_fields.mcast_write.noc_x_start +
            // packet_header.command_fields.mcast_write.mcast_rect_size_x >> 36) & 0x3F) <= 36);
            // ASSERT(((packet_header.command_fields.mcast_write.noc_y_start +
            // packet_header.command_fields.mcast_write.mcast_rect_size_y >> 42) & 0x3F) <= 36);
            break;

        case NocSendType::NOC_UNICAST_ATOMIC_INC:
            ASSERT(((packet_header->command_fields.unicast_seminc.noc_address >> 36) & 0x3F) <= 36);
            ASSERT(((packet_header->command_fields.unicast_seminc.noc_address >> 42) & 0x3F) <= 36);
            break;

        case NocSendType::NOC_UNICAST_INLINE_WRITE:
            ASSERT(((packet_header->command_fields.unicast_inline_write.noc_address >> 36) & 0x3F) <= 36);
            ASSERT(((packet_header->command_fields.unicast_inline_write.noc_address >> 42) & 0x3F) <= 36);
            break;

        case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC:
            ASSERT(((packet_header->command_fields.unicast_seminc_fused.noc_address >> 36) & 0x3F) <= 36);
            ASSERT(((packet_header->command_fields.unicast_seminc_fused.noc_address >> 42) & 0x3F) <= 36);
            ASSERT(((packet_header->command_fields.unicast_seminc_fused.semaphore_noc_address >> 36) & 0x3F) <= 36);
            ASSERT(((packet_header->command_fields.unicast_seminc_fused.semaphore_noc_address >> 42) & 0x3F) <= 36);
            break;

        default: ASSERT(false);
    }
}

FORCE_INLINE void validate(const LowLatencyPacketHeader& packet_header) {}
FORCE_INLINE bool is_valid(const LowLatencyPacketHeader& packet_header) {
    return (packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

}  // namespace tt::tt_fabric
