// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_op.hpp"

#include "convert_to_chw_program_factory.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::experimental::cnn {

void ConvertToCHW::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    TT_FATAL(input_tensors.size() == 1, "Expected 1 input tensor");

    TT_FATAL(this->memory_config.is_sharded(), "Output tensor must be sharded");
    // TT_FATAL(
    // this->memory_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED, "Output tensor must be width sharded");
    // C <= TILE_SIZE
    // HW % TILE_SIZE == 0
}

std::vector<tt::tt_metal::LegacyShape> ConvertToCHW::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& shape = input_tensors.at(0).get_logical_shape();
    const auto B = shape[0];
    const auto HW = shape[2];
    const auto C = shape[3];
    return {LegacyShape({B, 1, C, HW}, {B, 1, C, HW})};
}

std::vector<Tensor> ConvertToCHW::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->dtype, Layout::ROW_MAJOR, this->memory_config);
}

operation::ProgramWithCallbacks ConvertToCHW::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    auto& output = output_tensors.at(0);
    auto device_compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    return detail::multi_core_convert_to_chw(a, output, device_compute_with_storage_grid_size);
}
}  // namespace ttnn::operations::experimental::cnn
