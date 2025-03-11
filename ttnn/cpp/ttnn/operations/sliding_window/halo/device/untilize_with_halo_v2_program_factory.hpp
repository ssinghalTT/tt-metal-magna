// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks untilize_with_halo_multi_core_v2(
    Program& program,
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t ncores_c,
    const uint32_t max_out_nsticks_per_core,
    const Tensor& padding_config,
    const Tensor& gather_config0,
    const Tensor& gather_config1,
    const std::vector<uint16_t>& number_of_blocks_per_core,
    const bool remote_read,
    const bool transpose_mcast,
    Tensor& output_tensor,
    const int block_size,
    const bool capture_buffers,
    std::optional<std::reference_wrapper<const Tensor>> remote_temp,
    bool in_place);

}  // namespace ttnn::operations::data_movement::detail
