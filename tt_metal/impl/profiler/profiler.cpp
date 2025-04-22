// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dev_msgs.h>
#include <device.hpp>
#include <host_api.hpp>
#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <tracy/TracyTTDevice.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>

#include "assert.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "hostdevcommon/profiler_common.h"
#include "llrt.hpp"
#include "logger.hpp"
#include "metal_soc_descriptor.h"
#include "profiler.hpp"
#include "profiler_paths.hpp"
#include "profiler_state.hpp"
#include "tools/profiler/event_metadata.hpp"
#include "tracy/Tracy.hpp"
#include "tt_backend_api_types.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/xy_pair.h>

#include "fabric_edm_packet_header.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "tt_cluster.hpp"

namespace tt {

namespace tt_metal {

static kernel_profiler::PacketTypes get_packet_type(uint32_t timer_id) {
    return static_cast<kernel_profiler::PacketTypes>((timer_id >> 16) & 0x7);
}

void DeviceProfiler::readRiscProfilerResults(
    IDevice* device,
    const CoreCoord& worker_core,
    const std::optional<ProfilerOptionalMetadata>& metadata,
    std::ofstream& log_file_ofs,
    nlohmann::ordered_json& noc_trace_json_log) {
    ZoneScoped;
    chip_id_t device_id = device->id();

    HalProgrammableCoreType CoreType;
    int riscCount;

    if (tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(worker_core, device_id)) {
        CoreType = HalProgrammableCoreType::TENSIX;
        riscCount = 5;
    } else {
        auto active_eth_cores =
            tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(device_id);
        bool is_active_eth_core =
            active_eth_cores.find(
                tt::tt_metal::MetalContext::instance().get_cluster().get_logical_ethernet_core_from_virtual(
                    device_id, worker_core)) != active_eth_cores.end();

        CoreType = is_active_eth_core ? tt_metal::HalProgrammableCoreType::ACTIVE_ETH
                                      : tt_metal::HalProgrammableCoreType::IDLE_ETH;

        riscCount = 1;
    }
    profiler_msg_t* profiler_msg =
        MetalContext::instance().hal().get_dev_addr<profiler_msg_t*>(CoreType, HalL1MemAddrType::PROFILER);

    uint32_t coreFlatID =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_routing_to_profiler_flat_id(device_id).at(
            worker_core);
    uint32_t startIndex = coreFlatID * MAX_RISCV_PER_CORE * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC;

    std::vector<std::uint32_t> control_buffer = tt::llrt::read_hex_vec_from_core(
        device_id,
        worker_core,
        reinterpret_cast<uint64_t>(profiler_msg->control_vector),
        kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);

    if ((control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_BR_ER] == 0) &&
        (control_buffer[kernel_profiler::HOST_BUFFER_END_INDEX_NC] == 0)) {
        return;
    }

    // helper function to lookup opname from runtime id if metadata is available
    auto getOpNameIfAvailable = [&metadata](auto device_id, auto runtime_id) {
        return (metadata.has_value()) ? metadata->get_op_name(device_id, runtime_id) : "";
    };

    // translate worker core virtual coord to phys coordinates
    auto phys_coord = getPhysicalAddressFromVirtual(device_id, worker_core);

    int riscNum = 0;
    for (int riscEndIndex = 0; riscEndIndex < riscCount; riscEndIndex++) {
        uint32_t bufferEndIndex = control_buffer[riscEndIndex];
        uint32_t riscType;
        if (CoreType == HalProgrammableCoreType::TENSIX) {
            riscType = riscEndIndex;
        } else {
            riscType = 5;
        }
        if (bufferEndIndex > 0) {
            uint32_t bufferRiscShift = riscNum * PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC + startIndex;
            if ((control_buffer[kernel_profiler::DROPPED_ZONES] >> riscEndIndex) & 1) {
                std::string warningMsg = fmt::format(
                    "Profiler DRAM buffers were full, markers were dropped! device {}, worker core {}, {}, Risc {},  "
                    "bufferEndIndex = {}",
                    device_id,
                    worker_core.x,
                    worker_core.y,
                    tracy::riscName[riscEndIndex],
                    bufferEndIndex);
                TracyMessageC(warningMsg.c_str(), warningMsg.size(), tracy::Color::Tomato3);
                log_warning(warningMsg.c_str());
            }

            uint32_t riscNumRead = 0;
            uint32_t coreFlatIDRead = 0;
            uint32_t runCounterRead = 0;
            uint32_t runHostCounterRead = 0;

            bool newRunStart = false;

            uint32_t opTime_H = 0;
            uint32_t opTime_L = 0;
            std::string opname;
            for (int index = bufferRiscShift; index < (bufferRiscShift + bufferEndIndex);
                 index += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE) {
                if (!newRunStart && profile_buffer[index] == 0 && profile_buffer[index + 1] == 0) {
                    newRunStart = true;
                    opTime_H = 0;
                    opTime_L = 0;
                } else if (newRunStart) {
                    newRunStart = false;

                    // TODO(MO): Cleanup magic numbers
                    riscNumRead = profile_buffer[index] & 0x7;
                    coreFlatIDRead = (profile_buffer[index] >> 3) & 0xFF;
                    runCounterRead = profile_buffer[index + 1] & 0xFFFF;
                    runHostCounterRead = (profile_buffer[index + 1] >> 16) & 0xFFFF;

                    opname = getOpNameIfAvailable(device_id, runHostCounterRead);

                } else {
                    uint32_t timer_id = (profile_buffer[index] >> 12) & 0x7FFFF;
                    kernel_profiler::PacketTypes packet_type = get_packet_type(timer_id);

                    switch (packet_type) {
                        case kernel_profiler::ZONE_START:
                        case kernel_profiler::ZONE_END: {
                            uint32_t time_H = profile_buffer[index] & 0xFFF;
                            if (timer_id || time_H) {
                                uint32_t time_L = profile_buffer[index + 1];

                                if (opTime_H == 0) {
                                    opTime_H = time_H;
                                }
                                if (opTime_L == 0) {
                                    opTime_L = time_L;
                                }

                                TT_ASSERT(
                                    riscNumRead == riscNum,
                                    "Unexpected risc id, expected {}, read {}. In core {},{} at run {}",
                                    riscNum,
                                    riscNumRead,
                                    worker_core.x,
                                    worker_core.y,
                                    runCounterRead);
                                TT_ASSERT(
                                    coreFlatIDRead == coreFlatID,
                                    "Unexpected core id, expected {}, read {}. In core {},{} at run {}",
                                    coreFlatID,
                                    coreFlatIDRead,
                                    worker_core.x,
                                    worker_core.y,
                                    runCounterRead);

                                logPacketData(
                                    log_file_ofs,
                                    noc_trace_json_log,
                                    runCounterRead,
                                    runHostCounterRead,
                                    opname,
                                    device_id,
                                    phys_coord,
                                    coreFlatID,
                                    riscType,
                                    0,
                                    timer_id,
                                    (uint64_t(time_H) << 32) | time_L);
                            }
                        } break;
                        case kernel_profiler::ZONE_TOTAL: {
                            uint32_t sum = profile_buffer[index + 1];

                            uint32_t time_H = opTime_H;
                            uint32_t time_L = opTime_L;
                            logPacketData(
                                log_file_ofs,
                                noc_trace_json_log,
                                runCounterRead,
                                runHostCounterRead,
                                opname,
                                device_id,
                                phys_coord,
                                coreFlatID,
                                riscType,
                                sum,
                                timer_id,
                                (uint64_t(time_H) << 32) | time_L);

                            break;
                        }
                        case kernel_profiler::TS_DATA: {
                            uint32_t time_H = profile_buffer[index] & 0xFFF;
                            uint32_t time_L = profile_buffer[index + 1];
                            index += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
                            uint32_t data_H = profile_buffer[index];
                            uint32_t data_L = profile_buffer[index + 1];
                            logPacketData(
                                log_file_ofs,
                                noc_trace_json_log,
                                runCounterRead,
                                runHostCounterRead,
                                opname,
                                device_id,
                                phys_coord,
                                coreFlatID,
                                riscType,
                                (uint64_t(data_H) << 32) | data_L,
                                timer_id,
                                (uint64_t(time_H) << 32) | time_L);
                            continue;
                        }
                        case kernel_profiler::TS_EVENT: {
                            uint32_t time_H = profile_buffer[index] & 0xFFF;
                            uint32_t time_L = profile_buffer[index + 1];
                            logPacketData(
                                log_file_ofs,
                                noc_trace_json_log,
                                runCounterRead,
                                runHostCounterRead,
                                opname,
                                device_id,
                                phys_coord,
                                coreFlatID,
                                riscType,
                                0,
                                timer_id,
                                (uint64_t(time_H) << 32) | time_L);
                        }
                    }
                }
            }
        }
        riscNum++;
    }

    std::vector<uint32_t> control_buffer_reset(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
    control_buffer_reset[kernel_profiler::DRAM_PROFILER_ADDRESS] =
        control_buffer[kernel_profiler::DRAM_PROFILER_ADDRESS];
    control_buffer_reset[kernel_profiler::FLAT_ID] = control_buffer[kernel_profiler::FLAT_ID];
    control_buffer_reset[kernel_profiler::CORE_COUNT_PER_DRAM] = control_buffer[kernel_profiler::CORE_COUNT_PER_DRAM];

    tt::llrt::write_hex_vec_to_core(
        device_id, worker_core, control_buffer_reset, reinterpret_cast<uint64_t>(profiler_msg->control_vector));
}

void DeviceProfiler::firstTimestamp(uint64_t timestamp) {
    if (timestamp < smallest_timestamp) {
        smallest_timestamp = timestamp;
    }
}

void DeviceProfiler::logPacketData(
    std::ofstream& log_file_ofs,
    nlohmann::ordered_json& noc_trace_json_log,
    uint32_t run_id,
    uint32_t run_host_id,
    const std::string& opname,
    chip_id_t device_id,
    CoreCoord core,
    int /*core_flat*/,
    int risc_num,
    uint64_t data,
    uint32_t timer_id,
    uint64_t timestamp) {
    kernel_profiler::PacketTypes packet_type = get_packet_type(timer_id);
    uint32_t t_id = timer_id & 0xFFFF;
    std::string zone_name = "";
    std::string source_file = "";
    uint64_t source_line = 0;

    nlohmann::json metaData;

    if (hash_to_zone_src_locations.find((uint16_t)timer_id) != hash_to_zone_src_locations.end()) {
        std::stringstream source_info(hash_to_zone_src_locations[timer_id]);
        getline(source_info, zone_name, ',');
        getline(source_info, source_file, ',');

        std::string source_line_str;
        getline(source_info, source_line_str, ',');
        source_line = stoi(source_line_str);
    }

    if ((packet_type == kernel_profiler::ZONE_START) || (packet_type == kernel_profiler::ZONE_END)) {
        tracy::TTDeviceEventPhase zone_phase = tracy::TTDeviceEventPhase::begin;
        if (packet_type == kernel_profiler::ZONE_END) {
            zone_phase = tracy::TTDeviceEventPhase::end;
        }

        // TODO(MO) Until #14847 avoid attaching opID as the zone function name except for B and E FW
        // This is to avoid generating 5 to 10 times more source locations which is capped at 32K
        uint32_t tracy_run_host_id = run_host_id;
        if (zone_name.find("BRISC-FW") == std::string::npos && zone_name.find("ERISC-FW") == std::string::npos) {
            tracy_run_host_id = 0;
        }

        tracy::TTDeviceEvent event = tracy::TTDeviceEvent(
            tracy_run_host_id,
            device_id,
            core.x,
            core.y,
            risc_num,
            timer_id,
            timestamp,
            source_line,
            source_file,
            zone_name,
            zone_phase);

        auto ret = device_events.insert(event);
        this->current_zone_it = ret.first;
        event.run_num = 1;

        if (!ret.second) {
            return;
        }
        // Reset the command subtype, in case it isn't set during the command.
        this->current_dispatch_meta_data.cmd_subtype = "";
    }

    if (packet_type == kernel_profiler::TS_DATA) {
        if (this->current_zone_it != device_events.end()) {
            // Check if we are in BRISC Dispatch zone. If so, we could have gotten dispatch meta data packets
            // These packets can amend parent zone's info
            if (tracy::riscName[risc_num] == "BRISC" &&
                this->current_zone_it->zone_phase == tracy::TTDeviceEventPhase::begin &&
                this->current_zone_it->zone_name.find("DISPATCH") != std::string::npos) {
                if (zone_name.find("process_cmd") != std::string::npos) {
                    this->current_dispatch_meta_data.cmd_type =
                        fmt::format("{}", magic_enum::enum_name((CQDispatchCmdId)data));
                    metaData["dispatch_command_type"] = this->current_dispatch_meta_data.cmd_type;
                } else if (zone_name.find("runtime_host_id_dispatch") != std::string::npos) {
                    this->current_dispatch_meta_data.worker_runtime_id = (uint32_t)data;
                    metaData["workers_runtime_id"] = this->current_dispatch_meta_data.worker_runtime_id;
                } else if (zone_name.find("packed_data_dispatch") != std::string::npos) {
                    this->current_dispatch_meta_data.cmd_subtype = fmt::format(
                        "{}{}",
                        data & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST ? "MCAST," : "",
                        magic_enum::enum_name(static_cast<CQDispatchCmdPackedWriteType>(
                            (data >> 1) << CQ_DISPATCH_CMD_PACKED_WRITE_TYPE_SHIFT)));
                    metaData["dispatch_command_subtype"] = this->current_dispatch_meta_data.cmd_subtype;
                } else if (zone_name.find("packed_large_data_dispatch") != std::string::npos) {
                    this->current_dispatch_meta_data.cmd_subtype =
                        fmt::format("{}", magic_enum::enum_name(static_cast<CQDispatchCmdPackedWriteLargeType>(data)));
                    metaData["dispatch_command_subtype"] = this->current_dispatch_meta_data.cmd_subtype;
                }
                std::string cmd_name = this->current_dispatch_meta_data.cmd_subtype != ""
                                           ? this->current_dispatch_meta_data.cmd_subtype
                                           : this->current_dispatch_meta_data.cmd_type;
                tracy::TTDeviceEvent event = tracy::TTDeviceEvent(
                    this->current_dispatch_meta_data.worker_runtime_id,
                    this->current_zone_it->chip_id,
                    this->current_zone_it->core_x,
                    this->current_zone_it->core_y,
                    this->current_zone_it->risc,
                    this->current_zone_it->marker,
                    this->current_zone_it->timestamp,
                    this->current_zone_it->line,
                    this->current_zone_it->file,
                    fmt::format("{}:{}", this->current_dispatch_meta_data.worker_runtime_id, cmd_name),
                    this->current_zone_it->zone_phase);
                device_events.erase(this->current_zone_it);
                auto ret = device_events.insert(event);
                this->current_zone_it = ret.first;
            }
        }
    }

    firstTimestamp(timestamp);

    logPacketDataToCSV(
        log_file_ofs,
        device_id,
        core.x,
        core.y,
        tracy::riscName[risc_num],
        t_id,
        timestamp,
        data,
        run_id,
        run_host_id,
        opname,
        zone_name,
        packet_type,
        source_line,
        source_file,
        metaData);

    logNocTracePacketDataToJson(
        noc_trace_json_log,
        device_id,
        core.x,
        core.y,
        tracy::riscName[risc_num],
        t_id,
        timestamp,
        data,
        run_id,
        run_host_id,
        opname,
        zone_name,
        packet_type,
        source_line,
        source_file);
}

void DeviceProfiler::logPacketDataToCSV(
    std::ofstream& log_file_ofs,
    chip_id_t device_id,
    int core_x,
    int core_y,
    const std::string_view risc_name,
    uint32_t timer_id,
    uint64_t timestamp,
    uint64_t data,
    uint32_t run_id,
    uint32_t run_host_id,
    const std::string_view /*opname*/,
    const std::string_view zone_name,
    kernel_profiler::PacketTypes packet_type,
    uint64_t source_line,
    const std::string_view source_file,
    const nlohmann::json& metaData) {
    std::string metaDataStr = "";
    if (!metaData.is_null()) {
        metaDataStr = metaData.dump();
        std::replace(metaDataStr.begin(), metaDataStr.end(), ',', ';');
    }

    log_file_ofs << fmt::format(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                        device_id,
                        core_x,
                        core_y,
                        risc_name,
                        timer_id,
                        timestamp,
                        data,
                        run_id,
                        run_host_id,
                        zone_name,
                        magic_enum::enum_name(packet_type),
                        source_line,
                        source_file,
                        metaDataStr)
                 << std::endl;
}

void DeviceProfiler::logNocTracePacketDataToJson(
    nlohmann::ordered_json& noc_trace_json_log,
    chip_id_t device_id,
    int core_x,
    int core_y,
    const std::string_view risc_name,
    uint32_t /*timer_id*/,
    uint64_t timestamp,
    uint64_t data,
    uint32_t run_id,
    uint32_t run_host_id,
    const std::string_view opname,
    const std::string_view zone_name,
    kernel_profiler::PacketTypes packet_type,
    uint64_t /*source_line*/,
    const std::string_view /*source_file*/) {
    if (packet_type == kernel_profiler::ZONE_START || packet_type == kernel_profiler::ZONE_END) {
        if ((risc_name == "NCRISC" || risc_name == "BRISC") &&
            (zone_name.starts_with("TRUE-KERNEL-END") || zone_name.ends_with("-KERNEL"))) {
            tracy::TTDeviceEventPhase zone_phase = (packet_type == kernel_profiler::ZONE_END)
                                                       ? tracy::TTDeviceEventPhase::end
                                                       : tracy::TTDeviceEventPhase::begin;
            noc_trace_json_log.push_back(nlohmann::ordered_json{
                {"run_id", run_id},
                {"run_host_id", run_host_id},
                {"op_name", opname},
                {"proc", risc_name},
                {"zone", zone_name},
                {"zone_phase", magic_enum::enum_name(zone_phase)},
                {"sx", core_x},
                {"sy", core_y},
                {"timestamp", timestamp},
            });
        }

    } else if (packet_type == kernel_profiler::TS_DATA) {
        using EMD = KernelProfilerNocEventMetadata;
        EMD ev_md(data);
        std::variant<EMD::LocalNocEvent, EMD::FabricNoCEvent> ev_md_contents = ev_md.getContents();
        if (std::holds_alternative<EMD::LocalNocEvent>(ev_md_contents)) {
            auto local_noc_event = std::get<EMD::LocalNocEvent>(ev_md_contents);

            // NOTE: assume here that src and dest device_id are local;
            // serialization will coalesce and update to correct destination
            // based on fabric events
            nlohmann::ordered_json data = {
                {"run_id", run_id},
                {"run_host_id", run_host_id},
                {"op_name", opname},
                {"proc", risc_name},
                {"noc", magic_enum::enum_name(local_noc_event.noc_type)},
                {"vc", int(local_noc_event.noc_vc)},
                {"src_device_id", device_id},
                {"sx", core_x},
                {"sy", core_y},
                {"dst_device_id", device_id},
                {"num_bytes", local_noc_event.getNumBytes()},
                {"type", magic_enum::enum_name(ev_md.noc_xfer_type)},
                {"timestamp", timestamp},
            };

            // handle dst coordinates correctly for different NocEventType
            if (local_noc_event.dst_x == -1 || local_noc_event.dst_y == -1 ||
                ev_md.noc_xfer_type == EMD::NocEventType::READ_WITH_STATE ||
                ev_md.noc_xfer_type == EMD::NocEventType::WRITE_WITH_STATE) {
                // DO NOT emit destination coord; it isn't meaningful

            } else if (ev_md.noc_xfer_type == EMD::NocEventType::WRITE_MULTICAST) {
                auto phys_start_coord =
                    getPhysicalAddressFromVirtual(device_id, {local_noc_event.dst_x, local_noc_event.dst_y});
                data["mcast_start_x"] = phys_start_coord.x;
                data["mcast_start_y"] = phys_start_coord.y;
                auto phys_end_coord = getPhysicalAddressFromVirtual(
                    device_id, {local_noc_event.mcast_end_dst_x, local_noc_event.mcast_end_dst_y});
                data["mcast_end_x"] = phys_end_coord.x;
                data["mcast_end_y"] = phys_end_coord.y;
            } else {
                auto phys_coord =
                    getPhysicalAddressFromVirtual(device_id, {local_noc_event.dst_x, local_noc_event.dst_y});
                data["dx"] = phys_coord.x;
                data["dy"] = phys_coord.y;
            }

            noc_trace_json_log.push_back(std::move(data));
        } else {
            EMD::FabricNoCEvent fabric_noc_event = std::get<EMD::FabricNoCEvent>(ev_md_contents);
            auto phys_coord =
                getPhysicalAddressFromVirtual(device_id, {fabric_noc_event.dst_x, fabric_noc_event.dst_y});
            noc_trace_json_log.push_back(nlohmann::ordered_json{
                {"run_id", run_id},
                {"run_host_id", run_host_id},
                {"op_name", opname},
                {"proc", risc_name},
                {"sx", core_x},
                {"sy", core_y},
                {"type", magic_enum::enum_name(ev_md.noc_xfer_type)},
                {"dx", phys_coord.x},
                {"dy", phys_coord.y},
                {"routing_hops", fabric_noc_event.routing_hops},
                {"timestamp", timestamp},
            });
        }
    }
}

void DeviceProfiler::emitCSVHeader(
    std::ofstream& log_file_ofs, const tt::ARCH& device_architecture, int device_core_frequency) const {
    log_file_ofs << "ARCH: " << get_string_lowercase(device_architecture)
                 << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
    log_file_ofs << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run ID, "
                    "run host ID,  zone name, type, source line, source file, meta data"
                 << std::endl;
}

class FabricRoutingLookup {
public:
    using HopCount = int;
    using EthCoreToChannelMap =
        std::map<std::tuple<tt::tt_fabric::mesh_id_t, chip_id_t, CoreCoord>, tt::tt_fabric::chan_id_t>;
    using RoutingHopAndCoreCoordToDestinationMap = std::map<
        std::tuple<tt::tt_fabric::mesh_id_t, chip_id_t, CoreCoord, HopCount>,
        std::pair<tt::tt_fabric::mesh_id_t, chip_id_t>>;

    // Constructor takes ownership via move
    FabricRoutingLookup(EthCoreToChannelMap&& eth_map, RoutingHopAndCoreCoordToDestinationMap&& dest_map) :
        eth_core_to_channel_lookup_(std::move(eth_map)),
        routing_hop_and_core_coord_to_destination_(std::move(dest_map)) {}

    // Default constructor for cases where lookup is not built (e.g., non-1D fabric)
    FabricRoutingLookup() = default;

    // lookup APIs
    std::optional<tt::tt_fabric::chan_id_t> getEthCoreToChannelLookup(
        tt::tt_fabric::mesh_id_t mesh_id, chip_id_t chip_id, CoreCoord core_coord) const {
        auto it = eth_core_to_channel_lookup_.find(std::make_tuple(mesh_id, chip_id, core_coord));
        if (it != eth_core_to_channel_lookup_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::optional<std::pair<tt::tt_fabric::mesh_id_t, chip_id_t>> getRoutingHopAndCoreCoordToDestination(
        tt::tt_fabric::mesh_id_t mesh_id, chip_id_t chip_id, CoreCoord core_coord, HopCount hops) const {
        log_info(
            "getRoutingHopAndCoreCoordToDestination: mesh_id={}, chip_id={}, core_coord={}, hops={}",
            mesh_id,
            chip_id,
            core_coord,
            hops);
        auto it = routing_hop_and_core_coord_to_destination_.find(std::make_tuple(mesh_id, chip_id, core_coord, hops));
        if (it != routing_hop_and_core_coord_to_destination_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    EthCoreToChannelMap eth_core_to_channel_lookup_;
    RoutingHopAndCoreCoordToDestinationMap routing_hop_and_core_coord_to_destination_;
};

FabricRoutingLookup DeviceProfiler::buildFabricRoutingLookup() const {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // establish that we have a 1D fabric, otherwise bail
    tt::tt_metal::FabricConfig fabric_config = cluster.get_fabric_config();
    if (!tt::tt_fabric::is_1d_fabric_config(fabric_config)) {
        log_info("Skipping fabric routing lookup build; topology is not 1D Ring or Linear");
        // Return default-constructed (empty) lookup object
        return FabricRoutingLookup{};
    }

    const tt::tt_fabric::ControlPlane* control_plane = cluster.get_control_plane();
    TT_ASSERT(control_plane != nullptr);

    // assume their is only one mesh (for now)
    tt::tt_fabric::mesh_id_t mesh_id = 0;

    auto user_exposed_chip_ids = cluster.user_exposed_chip_ids();
    std::vector<chip_id_t> sorted_user_exposed_chip_ids(user_exposed_chip_ids.begin(), user_exposed_chip_ids.end());
    std::sort(sorted_user_exposed_chip_ids.begin(), sorted_user_exposed_chip_ids.end());

    // Use the type aliases from the locally defined class
    FabricRoutingLookup::EthCoreToChannelMap eth_core_to_channel_lookup;
    FabricRoutingLookup::RoutingHopAndCoreCoordToDestinationMap routing_hop_and_core_coord_to_destination;

    for (chip_id_t chip_id_src : sorted_user_exposed_chip_ids) {
        for (chip_id_t chip_id_dst : sorted_user_exposed_chip_ids) {
            if (chip_id_src == chip_id_dst) {
                continue;
            }
            auto routers_to_chip = control_plane->get_routers_to_chip(mesh_id, chip_id_src, mesh_id, chip_id_dst);
            // log_info("src_chip_id: {}, dst_chip_id: {}", chip_id_src, chip_id_dst);
            for (const auto& [routing_plane_id, coord] : routers_to_chip) {
                auto valid_eth_chans =
                    control_plane->get_valid_eth_chans_on_routing_plane(mesh_id, chip_id_src, routing_plane_id);
                // log_info("    routing plane {} via eth core ({}, {})", routing_plane_id, coord.x, coord.y);
                for (const auto& chan_id : valid_eth_chans) {
                    eth_core_to_channel_lookup.emplace(std::make_tuple(mesh_id, chip_id_src, coord), chan_id);
                    auto route = control_plane->get_fabric_route(mesh_id, chip_id_src, mesh_id, chip_id_dst, chan_id);
                    FabricRoutingLookup::HopCount hops = route.size();  // Use type alias
                    routing_hop_and_core_coord_to_destination.emplace(
                        std::make_tuple(mesh_id, chip_id_src, coord, hops), std::make_pair(mesh_id, chip_id_dst));
                }
            }
        }
    }

    // Construct and return the lookup object, moving the maps into it
    return FabricRoutingLookup(
        std::move(eth_core_to_channel_lookup), std::move(routing_hop_and_core_coord_to_destination));
}

void DeviceProfiler::serializeJsonNocTraces(
    const nlohmann::ordered_json& noc_trace_json_log,
    const std::filesystem::path& output_dir,
    chip_id_t device_id,
    const FabricRoutingLookup& routing_lookup) {
    // create output directory if it does not exist
    std::filesystem::create_directories(output_dir);
    if (!std::filesystem::is_directory(output_dir)) {
        log_error(
            "Could not write noc event json trace to '{}' because the directory path could not be created!",
            output_dir);
        return;
    }

    // bin events by runtime id
    using RuntimeID = uint32_t;
    std::unordered_map<RuntimeID, nlohmann::json::array_t> events_by_opname;
    for (auto& json_event : noc_trace_json_log) {
        RuntimeID runtime_id = json_event.value("run_host_id", -1);
        events_by_opname[runtime_id].push_back(json_event);
    }

    // sort events in each opname group by proc first, then timestamp
    for (auto& [runtime_id, events] : events_by_opname) {
        std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
            auto sx_a = a.value("sx", 0);
            auto sy_a = a.value("sy", 0);
            auto sx_b = b.value("sx", 0);
            auto sy_b = b.value("sy", 0);
            auto proc_a = a.value("proc", "");
            auto proc_b = b.value("proc", "");
            auto timestamp_a = a.value("timestamp", 0);
            auto timestamp_b = b.value("timestamp", 0);
            return std::tie(sx_a, sy_a, proc_a, timestamp_a) < std::tie(sx_b, sy_b, proc_b, timestamp_b);
        });
    }

    // for each opname in events_by_opname, adjust timestamps to be relative to the smallest timestamp within the group
    // with identical sx,sy,proc
    for (auto& [runtime_id, events] : events_by_opname) {
        std::tuple<int, int, std::string> reference_event_loc;
        uint64_t reference_timestamp = 0;
        for (auto& event : events) {
            std::string zone = event.value("zone", "");
            std::string zone_phase = event.value("zone_phase", "");
            uint64_t curr_timestamp = event.value("timestamp", 0);
            // if -KERNEL::begin event is found, reset the reference timestamp
            if (zone.ends_with("-KERNEL") && zone_phase == "begin") {
                reference_timestamp = curr_timestamp;
            }

            // fix timestamp to be relative to reference_timestamp
            event["timestamp"] = curr_timestamp - reference_timestamp;
        }
    }

    constexpr tt::tt_fabric::mesh_id_t mesh_id = 0;  // Assuming single mesh

    std::unordered_map<RuntimeID, nlohmann::json::array_t> processed_events_by_opname;

    for (auto& [runtime_id, events] : events_by_opname) {
        nlohmann::json::array_t coalesced_events;
        for (size_t i = 0; i < events.size(); /* manual increment */) {
            const auto& current_event = events[i];
            bool coalesced = false;

            if (current_event.contains("type") && current_event["type"] == "FABRIC_UNICAST_WRITE" &&
                (i + 1 < events.size())) {
                const auto& next_event_const = events[i + 1];
                if (next_event_const.contains("type") && next_event_const["type"] == "WRITE_") {
                    // Check if timestamps are close enough; otherwise
                    double ts_diff = next_event_const.value("timestamp", 0.0) - current_event.value("timestamp", 0.0);
                    if (ts_diff > 1000) {
                        log_warning(
                            "Failed to coalesce fabric noc trace events because timestamps are implausibly far apart.");
                    } else {
                        try {
                            // router eth core location is derived from the original noc write event
                            int phys_eth_x = next_event_const.at("dx").get<int>();
                            int phys_eth_y = next_event_const.at("dy").get<int>();

                            const metal_SocDescriptor& soc_desc =
                                tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
                            CoreCoord translated_eth_core_coord = soc_desc.translate_coord_to(
                                {(uint32_t)phys_eth_x, (uint32_t)phys_eth_y},
                                CoordSystem::PHYSICAL,
                                CoordSystem::TRANSLATED);

                            int hops = current_event.at("routing_hops").get<int>();
                            auto dest_info = routing_lookup.getRoutingHopAndCoreCoordToDestination(
                                mesh_id, device_id, translated_eth_core_coord, hops);

                            if (dest_info.has_value()) {
                                auto [dest_mesh_id, dest_chip_id] = dest_info.value();
                                nlohmann::ordered_json modified_write_event = next_event_const;
                                modified_write_event["dst_device_id"] = dest_chip_id;
                                modified_write_event["timestamp"] = current_event["timestamp"];

                                // replace original eth core destination with true destination
                                modified_write_event["dx"] = current_event["dx"];
                                modified_write_event["dy"] = current_event["dy"];

                                coalesced_events.push_back(std::move(modified_write_event));
                                coalesced = true;
                            } else {
                                log_warning(
                                    "Fabric routing lookup failed for event in op '{}' at ts {}: src_dev={}, "
                                    "eth_core=({}, {}), hops={}. Keeping original events.",
                                    current_event.value("op_name", "N/A"),
                                    current_event.value("timestamp", 0.0),
                                    device_id,
                                    translated_eth_core_coord.x,
                                    translated_eth_core_coord.y,
                                    hops);
                            }
                        } catch (const nlohmann::json::exception& e) {
                            log_warning(
                                "JSON parsing error during event coalescing for event in op '{}' at index {}: {}. "
                                "Keeping original events.",
                                current_event.value("op_name", "N/A"),
                                i,
                                e.what());
                        }
                    }
                } else {
                    log_info(
                        "noc event following fabric event is not a WRITE_, but instead : {}", current_event.dump(2));
                }
            }

            if (coalesced) {
                i += 2;  // Skip both original events
            } else {
                // If not coalesced or lookup failed, add the current event
                coalesced_events.push_back(current_event);
                i += 1;
            }
        }
        // Store the final coalesced/processed list for this op_name
        processed_events_by_opname[runtime_id] = std::move(coalesced_events);
    }

    log_info("Writing profiler noc traces to '{}'", output_dir);
    for (auto& [runtime_id, events] : processed_events_by_opname) {
        // dump events to a json file inside directory output_dir named after the opname
        std::filesystem::path rpt_path = output_dir;
        std::string op_name = events.front().value("op_name", "UnknownOP");
        if (!op_name.empty()) {
            rpt_path /= fmt::format("noc_trace_dev{}_{}_ID{}.json", device_id, op_name, runtime_id);
        } else {
            rpt_path /= fmt::format("noc_trace_dev{}_ID{}.json", device_id, runtime_id);
        }
        std::ofstream file(rpt_path);
        if (file.is_open()) {
            // Write the final processed events for this op
            file << nlohmann::json(std::move(events)).dump(2);
        } else {
            log_error("Could not open file '{}' for writing noc trace.", rpt_path);
        }
    }
}

CoreCoord DeviceProfiler::getPhysicalAddressFromVirtual(chip_id_t device_id, const CoreCoord& c) const {
    bool coord_is_translated = c.x >= MetalContext::instance().hal().get_virtual_worker_start_x() - 1 &&
                               c.y >= MetalContext::instance().hal().get_virtual_worker_start_y() - 1;
    try {
        if (device_architecture == tt::ARCH::WORMHOLE_B0 && coord_is_translated) {
            const metal_SocDescriptor& soc_desc =
                tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
            // disable linting here; slicing is __intended__
            // NOLINTBEGIN
            return soc_desc.translate_coord_to(c, CoordSystem::TRANSLATED, CoordSystem::PHYSICAL);
            // NOLINTEND
        } else {
            // tt:ARCH::BLACKHOLE currently doesn't have any translated coordinate adjustment
            return c;
        }
    } catch (const std::exception& e) {
        log_error("Failed to translate virtual coordinate {},{} to physical", c.x, c.y);
        return c;
    }
}

DeviceProfiler::DeviceProfiler(const bool new_logs) {
#if defined(TRACY_ENABLE)
    ZoneScopedC(tracy::Color::Green);
    output_dir = std::filesystem::path(get_profiler_logs_dir());
    std::filesystem::create_directories(output_dir);
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;

    if (new_logs) {
        std::filesystem::remove(log_path);
    }

    this->current_zone_it = device_events.begin();
#endif
}

DeviceProfiler::~DeviceProfiler() {
#if defined(TRACY_ENABLE)
    ZoneScoped;
    pushTracyDeviceResults();
    for (auto tracyCtx : device_tracy_contexts) {
        TracyTTDestroy(tracyCtx.second);
    }
#endif
}

void DeviceProfiler::freshDeviceLog() {
#if defined(TRACY_ENABLE)
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::filesystem::remove(log_path);
#endif
}

void DeviceProfiler::setOutputDir(const std::string& new_output_dir) {
#if defined(TRACY_ENABLE)
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
#endif
}

void DeviceProfiler::setDeviceArchitecture(tt::ARCH device_arch) {
#if defined(TRACY_ENABLE)
    device_architecture = device_arch;
#endif
}

uint32_t DeviceProfiler::hash32CT(const char* str, size_t n, uint32_t basis) {
    return n == 0 ? basis : hash32CT(str + 1, n - 1, (basis ^ str[0]) * UINT32_C(16777619));
}

uint16_t DeviceProfiler::hash16CT(const std::string& str) {
    uint32_t res = hash32CT(str.c_str(), str.length());
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}

void DeviceProfiler::generateZoneSourceLocationsHashes() {
    std::ifstream log_file(tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG);
    std::string line;
    while (std::getline(log_file, line)) {
        std::string delimiter = "'#pragma message: ";
        int delimiter_index = line.find(delimiter) + delimiter.length();
        std::string zone_src_location = line.substr(delimiter_index, line.length() - delimiter_index - 1);

        uint16_t hash_16bit = hash16CT(zone_src_location);

        auto did_insert = zone_src_locations.insert(zone_src_location);
        if (did_insert.second && (hash_to_zone_src_locations.find(hash_16bit) != hash_to_zone_src_locations.end())) {
            log_warning("Source location hashes are colliding, two different locations are having the same hash");
        }
        hash_to_zone_src_locations.emplace(hash_16bit, zone_src_location);
    }
}

void DeviceProfiler::dumpResults(
    IDevice* device,
    const std::vector<CoreCoord>& worker_cores,
    ProfilerDumpState state,
    const std::optional<ProfilerOptionalMetadata>& metadata) {
#if defined(TRACY_ENABLE)
    ZoneScoped;

    auto device_id = device->id();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    device_core_frequency = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device_id);

    generateZoneSourceLocationsHashes();

    FabricRoutingLookup routing_lookup = buildFabricRoutingLookup();

    if (output_dram_buffer != nullptr) {
        const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;
        if (USE_FAST_DISPATCH) {
            if (state == ProfilerDumpState::LAST_CLOSE_DEVICE) {
                if (rtoptions.get_profiler_do_dispatch_cores()) {
                    tt_metal::detail::ReadFromBuffer(output_dram_buffer, profile_buffer);
                }
            } else {
                EnqueueReadBuffer(device->command_queue(), output_dram_buffer, profile_buffer, true);
            }
        } else {
            if (state != ProfilerDumpState::LAST_CLOSE_DEVICE) {
                tt_metal::detail::ReadFromBuffer(output_dram_buffer, profile_buffer);
            }
        }

        if (rtoptions.get_profiler_noc_events_enabled()) {
            log_warning("Profiler NoC events are enabled; this can add 1-15% cycle overhead to typical operations!");
        }

        // open CSV log file
        std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
        std::ofstream log_file_ofs;

        // append to existing CSV log file if it already exists
        if (std::filesystem::exists(log_path)) {
            log_file_ofs.open(log_path, std::ios_base::app);
        } else {
            log_file_ofs.open(log_path);
            emitCSVHeader(log_file_ofs, device_architecture, device_core_frequency);
        }

        // create nlohmann json log object
        nlohmann::ordered_json noc_trace_json_log = nlohmann::json::array();

        if (!log_file_ofs) {
            log_error("Could not open kernel profiler dump file '{}'", log_path);
        } else {
            for (const auto& worker_core : worker_cores) {
                readRiscProfilerResults(device, worker_core, metadata, log_file_ofs, noc_trace_json_log);
            }

            // if defined, used profiler_noc_events_report_path to write json log. otherwise use output_dir
            auto rpt_path = rtoptions.get_profiler_noc_events_report_path();
            if (rpt_path.empty()) {
                rpt_path = output_dir;
            }

            // serialize noc traces only in normal state, to avoid overwriting individual trace files
            if (state == ProfilerDumpState::NORMAL && rtoptions.get_profiler_noc_events_enabled()) {
                serializeJsonNocTraces(noc_trace_json_log, rpt_path, device_id, routing_lookup);
            }
        }
    } else {
        log_warning("DRAM profiler buffer is not initialized");
    }
#endif
}

void DeviceProfiler::pushTracyDeviceResults() {
#if defined(TRACY_ENABLE)
    ZoneScoped;
    std::set<std::pair<uint32_t, CoreCoord>> device_cores_set;
    std::vector<std::pair<uint32_t, CoreCoord>> device_cores;
    for (auto& event : device_events) {
        std::pair<uint32_t, CoreCoord> device_core = {event.chip_id, (CoreCoord){event.core_x, event.core_y}};
        auto ret = device_cores_set.insert(device_core);
        if (ret.second) {
            device_cores.push_back(device_core);
        }
    }

    static double delay = 0;
    static double frequency = 0;
    static uint64_t cpuTime = 0;

    for (auto& device_core : device_cores) {
        chip_id_t device_id = device_core.first;
        CoreCoord worker_core = device_core.second;

        if (device_core_sync_info.find(worker_core) != device_core_sync_info.end()) {
            cpuTime = get<0>(device_core_sync_info.at(worker_core));
            delay = get<1>(device_core_sync_info.at(worker_core));
            frequency = get<2>(device_core_sync_info.at(worker_core));
            log_info(
                "Device {} sync info are, frequency {} GHz,  delay {} cycles and, sync point {} seconds",
                device_id,
                frequency,
                delay,
                cpuTime);
        }
    }

    for (auto& device_core : device_cores) {
        chip_id_t device_id = device_core.first;
        CoreCoord worker_core = device_core.second;

        if (delay == 0.0 || frequency == 0.0) {
            delay = smallest_timestamp;
            frequency = device_core_frequency / 1000.0;
            cpuTime = TracyGetCpuTime();
            log_warning(
                "For device {}, core {},{} default frequency was used and its zones will be out of sync",
                device_id,
                worker_core.x,
                worker_core.y);
        }

        if (device_tracy_contexts.find(device_core) == device_tracy_contexts.end()) {
            auto tracyCtx = TracyTTContext();
            std::string tracyTTCtxName =
                fmt::format("Device: {}, Core ({},{})", device_id, worker_core.x, worker_core.y);

            TracyTTContextPopulate(tracyCtx, cpuTime, delay, frequency);

            TracyTTContextName(tracyCtx, tracyTTCtxName.c_str(), tracyTTCtxName.size());

            device_tracy_contexts.emplace(device_core, tracyCtx);
        }
    }

    for (auto event : device_events) {
        std::pair<uint32_t, CoreCoord> device_core = {event.chip_id, (CoreCoord){event.core_x, event.core_y}};
        event.timestamp = event.timestamp * this->freqScale + this->shift;
        if (event.zone_phase == tracy::TTDeviceEventPhase::begin) {
            TracyTTPushStartZone(device_tracy_contexts[device_core], event);
        } else if (event.zone_phase == tracy::TTDeviceEventPhase::end) {
            TracyTTPushEndZone(device_tracy_contexts[device_core], event);
        }
    }
    device_events.clear();
#endif
}

bool getDeviceProfilerState() { return tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled(); }

}  // namespace tt_metal

}  // namespace tt
