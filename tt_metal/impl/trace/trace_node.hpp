// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "program/program_impl.hpp"

namespace tt::tt_metal {

struct TraceNode {
    std::shared_ptr<detail::ProgramImpl> program;
    uint32_t program_runtime_id;
    SubDeviceId sub_device_id;

    std::vector<std::vector<uint8_t>> rta_data;
    std::vector<std::vector<uint32_t>> cb_configs_payloads;
};

}  // namespace tt::tt_metal
