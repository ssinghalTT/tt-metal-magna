// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt_common/mailbox.hpp"
#define COMPILE_FOR_ERISC

#include "tt_align.hpp"
#include <dev_msgs.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "blackhole/bh_hal.hpp"
#include "core_config.h"
#include "dev_mem_map.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "noc/noc_parameters.h"
#include <umd/device/tt_core_coordinates.h>

#define GET_IERISC_MAILBOX_ADDRESS_HOST(x) ((std::uint64_t)&(((mailboxes_t*)MEM_IERISC_MAILBOX_BASE)->x))

namespace tt::tt_metal::blackhole {

HalCoreInfoType create_idle_eth_mem_map() {
    std::uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    static_assert(MEM_IERISC_MAP_END % L1_ALIGNMENT == 0);

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_ETH_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_IERISC_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_IERISC_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_IERISC_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = GET_IERISC_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_IERISC_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_IERISC_MAP_END;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        tt::align(MEM_AERISC_MAP_END + MEM_ERISC_KERNEL_CONFIG_SIZE, max_alignment);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_IERISC_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_IERISC_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_IERISC_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_IERISC_BANK_TO_NOC_SCRATCH;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX_MSG)] =
        MEM_SYSENG_ETH_MAILBOX_ADDR + offsetof(EthFwMailbox, msg);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX_ARG0)] =
        MEM_SYSENG_ETH_MAILBOX_ADDR + offsetof(EthFwMailbox, arg[0]);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX_ARG1)] =
        MEM_SYSENG_ETH_MAILBOX_ADDR + offsetof(EthFwMailbox, arg[1]);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX_ARG2)] =
        MEM_SYSENG_ETH_MAILBOX_ADDR + offsetof(EthFwMailbox, arg[2]);

    std::vector<std::uint32_t> mem_map_sizes;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_ETH_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_IERISC_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    // TODO: this is wrong, need eth specific value. For now use same value as idle
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_ERISC_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        MEM_ETH_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)];
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_IERISC_BANK_TO_NOC_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX_MSG)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX_ARG0)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX_ARG1)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX_ARG2)] = sizeof(uint32_t);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(NumEthDispatchClasses);
    std::vector<HalJitBuildConfig> processor_types(MaxDMProcessorsPerCoreType);
    for (std::uint8_t processor_class_idx = 0; processor_class_idx < NumEthDispatchClasses; processor_class_idx++) {
        processor_types[static_cast<std::size_t>(EthProcessorTypes::DM0)] = HalJitBuildConfig{
            .fw_base_addr = MEM_IERISC_FIRMWARE_BASE,
            .local_init_addr = MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH,
            .fw_launch_addr = IERISC_RESET_PC,
            .fw_launch_addr_value = MEM_IERISC_FIRMWARE_BASE};
        processor_types[static_cast<std::size_t>(EthProcessorTypes::DM1)] = HalJitBuildConfig{
            .fw_base_addr = MEM_SLAVE_IERISC_FIRMWARE_BASE,
            .local_init_addr = MEM_SLAVE_IERISC_INIT_LOCAL_L1_BASE_SCRATCH,
            .fw_launch_addr = SLAVE_IERISC_RESET_PC,
            .fw_launch_addr_value = MEM_SLAVE_IERISC_FIRMWARE_BASE};
        processor_classes[processor_class_idx] = processor_types;
    }

    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);
    fw_mailbox_addr[utils::underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_CALL)] = MEM_SYSENG_ETH_MSG_CALL;
    fw_mailbox_addr[utils::underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_DONE)] = MEM_SYSENG_ETH_MSG_DONE;
    fw_mailbox_addr[utils::underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_LINK_STATUS_CHECK)] =
        MEM_SYSENG_ETH_MSG_LINK_STATUS_CHECK;
    fw_mailbox_addr[utils::underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_RELEASE_CORE)] =
        MEM_SYSENG_ETH_MSG_RELEASE_CORE;

    // TODO: Review if this should  be 2 (the number of eth processors)
    // Hardcode to 1 to keep size as before
    static_assert(llrt_common::k_SingleProcessorMailboxSize<EthProcessorTypes> <= MEM_IERISC_MAILBOX_SIZE);
    return {
        HalProgrammableCoreType::IDLE_ETH,
        CoreType::ETH,
        processor_classes,
        mem_map_bases,
        mem_map_sizes,
        fw_mailbox_addr,
        false /*supports_cbs*/,
        false /*supports_receiving_multicast_cmds*/};
}

}  // namespace tt::tt_metal::blackhole
