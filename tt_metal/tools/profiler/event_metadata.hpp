// SPDX-FileCopyrightText: 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>  // for std::memcpy
#include <variant>  // Added include
#include <limits>
#include <algorithm>

struct alignas(uint64_t) KernelProfilerNocEventMetadata {
    enum class NocType : unsigned char { UNDEF = 0, NOC_0 = 1, NOC_1 = 2 };
    using NocVirtualChannel = int8_t;
    static constexpr uint32_t PAYLOAD_CHUNK_SIZE = 32;

    // New struct for local NOC events
    struct LocalNocEvent {
        int8_t dst_x;
        int8_t dst_y;
        int8_t mcast_end_dst_x;
        int8_t mcast_end_dst_y;
        NocType noc_type : 4;
        NocVirtualChannel noc_vc : 4;
        uint8_t payload_chunks;

        void setNumBytes(uint32_t num_bytes) {
            uint32_t bytes_rounded_up = (num_bytes + PAYLOAD_CHUNK_SIZE - 1) / PAYLOAD_CHUNK_SIZE;
            payload_chunks = std::min(uint32_t(std::numeric_limits<uint8_t>::max()), bytes_rounded_up);
        }
        uint32_t getNumBytes() const { return payload_chunks * PAYLOAD_CHUNK_SIZE; }
    };

    // Existing struct for fabric NOC events
    struct FabricNoCEvent {
        int8_t dst_x;
        int8_t dst_y;
        int8_t mcast_end_dst_x;
        int8_t mcast_end_dst_y;
        int8_t routing_fields;
        int8_t noc_send_type;
    };

    // Union to hold either local or fabric event data
    union EventData {
        LocalNocEvent local_event;
        FabricNoCEvent fabric_event;
    } data;

    // --- Type enum (tag) --- Must be defined before use in constructor
    enum class NocEventType : unsigned char {
        UNDEF = 0,
        READ,
        READ_SET_STATE,
        READ_SET_TRID,
        READ_WITH_STATE,
        READ_WITH_STATE_AND_TRID,
        READ_BARRIER_START,
        READ_BARRIER_END,
        READ_BARRIER_WITH_TRID,
        READ_DRAM_SHARDED_SET_STATE,
        READ_DRAM_SHARDED_WITH_STATE,

        WRITE_,
        WRITE_WITH_TRID,
        WRITE_INLINE,
        WRITE_MULTICAST,
        WRITE_SET_STATE,
        WRITE_WITH_STATE,
        WRITE_WITH_TRID_SET_STATE,
        WRITE_WITH_TRID_WITH_STATE,
        WRITE_BARRIER_START,
        WRITE_BARRIER_END,
        WRITE_BARRIER_WITH_TRID,
        WRITE_FLUSH,

        FULL_BARRIER,

        ATOMIC_BARRIER,
        SEMAPHORE_INC,
        SEMAPHORE_WAIT,
        SEMAPHORE_SET,

        FABRIC_NOC_EVENT,

        UNSUPPORTED
    };
    NocEventType noc_xfer_type;

    KernelProfilerNocEventMetadata() : data{.local_event = {}}, noc_xfer_type(NocEventType::UNDEF) {}

    // for deserialization on host side
    explicit KernelProfilerNocEventMetadata(const uint64_t raw_data) {
        std::memcpy(this, &raw_data, sizeof(KernelProfilerNocEventMetadata));
    }

    // Getter to return the correct variant based on the tag
    std::variant<LocalNocEvent, FabricNoCEvent> getContents() const {
        if (noc_xfer_type == NocEventType::FABRIC_NOC_EVENT) {
            return data.fabric_event;
        } else {
            return data.local_event;
        }
    }

    uint64_t asU64() const {
        uint64_t ret;
        std::memcpy(&ret, this, sizeof(uint64_t));
        return ret;
    }
};
static_assert(sizeof(KernelProfilerNocEventMetadata) == sizeof(uint64_t));
