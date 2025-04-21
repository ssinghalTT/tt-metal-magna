#pragma once

#include "event_metadata.hpp"
#include "noc_event_profiler.hpp"
#include "api/tt-metalium/fabric_edm_packet_header.hpp"

int get_routing_hops(const tt::tt_fabric::LowLatencyRoutingFields& llrf) {
    using namespace tt::tt_fabric;
    uint32_t value = llrf.value;
    uint32_t hops = 0;
    while (value) {
        value >>= LowLatencyRoutingFields::FIELD_WIDTH;
        hops++;
    }
    return hops;
}

void record_fabric_header(volatile tt::tt_fabric::LowLatencyPacketHeader* fabric_header_ptr) {
    using namespace tt::tt_fabric;
    LowLatencyPacketHeader* fh = const_cast<LowLatencyPacketHeader*>(fabric_header_ptr);

    auto noc_send_type = fh->get_noc_send_type();
    if (noc_send_type == NocSendType::NOC_UNICAST_WRITE) {
        const auto& unicast_write_cmd = fh->get_command_fields().unicast_write;
        noc_event_profiler::recordFabricNocEvent(
            KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_WRITE,
            unicast_write_cmd.noc_address,
            get_routing_hops(fh->routing_fields));
    }
}

#define RECORD_FABRIC_HEADER(_fabric_header_ptr)  \
    {                                             \
        record_fabric_header(_fabric_header_ptr); \
    }
