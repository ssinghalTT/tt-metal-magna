#pragma once

#include "noc_event_profiler.hpp"
#include "api/tt-metalium/fabric_edm_packet_header.hpp"

namespace kernel_profiler {

int get_low_latency_routing_hops(uint32_t llrf_value) {
    uint32_t value = llrf_value;
    uint32_t hops = 0;
    while (value) {
        value >>= tt::tt_fabric::LowLatencyRoutingFields::FIELD_WIDTH;
        hops++;
    }
    return hops;
}

int get_routing_hops(uint8_t routing_fields_value) {
    return tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK & routing_fields_value;
}

void record_fabric_header(const volatile tt::tt_fabric::PacketHeader* fabric_header_ptr) {
#ifdef PROFILE_NOC_EVENTS

#if defined FABRIC_LOW_LATENCY_MODE and FABRIC_LOW_LATENCY_MODE == 1
    auto fh = reinterpret_cast<const volatile tt::tt_fabric::LowLatencyPacketHeader*>(fabric_header_ptr);
    auto num_hops = get_low_latency_routing_hops(fh->routing_fields.value);
#else
    auto fh = reinterpret_cast<const volatile tt::tt_fabric::PacketHeader*>(fabric_header_ptr);
    auto num_hops = get_routing_hops(fh->routing_fields.value);
#endif

    auto noc_send_type = fh->get_noc_send_type();

    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const volatile auto& unicast_write_cmd = fh->get_command_fields().unicast_write;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_WRITE,
                unicast_write_cmd.noc_address,
                num_hops);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            const volatile auto& unicast_write_cmd = fh->get_command_fields().unicast_seminc;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_ATOMIC_INC,
                unicast_write_cmd.noc_address,
                num_hops);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const volatile auto& unicast_write_cmd = fh->get_command_fields().unicast_seminc_fused;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_FUSED_UNICAST_ATOMIC_INC,
                unicast_write_cmd.noc_address,
                num_hops);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {
            const volatile auto& unicast_write_cmd = fh->get_command_fields().unicast_inline_write;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_INLINE_WRITE,
                unicast_write_cmd.noc_address,
                num_hops);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE:
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_ATOMIC_INC: {
            // unsupported for now; metadata exceeds packet size
            break;
        }
    }
#endif
}
}  // namespace kernel_profiler

#define RECORD_FABRIC_HEADER(_fabric_header_ptr)                                                \
    {                                                                                           \
        kernel_profiler::record_fabric_header(                                                  \
            reinterpret_cast<const volatile tt::tt_fabric::PacketHeader*>(_fabric_header_ptr)); \
    }
