// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_op.hpp"
#include "argmax_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction {

void ArgMax::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");

    TT_FATAL(this->output_dtype == DataType::UINT32, "Only UINT32 is supported for outputs!");
    TT_FATAL(
        this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for outputs!");

    TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");
    const auto& optional_output_tensor = output_tensors.at(0);
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            optional_output_tensor.value().get_dtype() == DataType::UINT32, "Only UINT32 is supported for outputs!");
        TT_FATAL(
            optional_output_tensor.value().memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Only INTERLEAVED memory layout is supported for outputs!");
    }

    if (this->dim.has_value()) {
        const uint32_t input_rank = input_tensor_a.get_padded_shape().rank();
        const uint32_t normalized_dim = dim.value() < 0 ? dim.value() + input_rank : dim.value();

        // TODO: Add support for normalized_dim = 0, 1, 2
        TT_FATAL(normalized_dim == (input_rank - 1), "Only argmax on last dim is supported!");
    }

    auto input_shape = input_tensor_a.get_padded_shape();
    TT_FATAL(input_shape[0] == 1, "dim 0 must be 1");
    TT_FATAL(input_shape[1] == 1, "dim 1 must be 1");
}

std::vector<TensorSpec> ArgMax::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0)->get_tensor_spec()};
    }

    const auto& input_tensor = input_tensors[0];
    ttnn::Shape output_shape({1, 1, 1, 1});
    if (this->dim.has_value()) {
        auto input_shape = input_tensors[0].get_logical_shape();
        output_shape = ttnn::Shape{input_shape[0], input_shape[1], 1, input_shape[2]};
    }
    return {
        TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(input_tensor.get_layout()), output_mem_config))};
}

std::vector<Tensor> ArgMax::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors[0].device())};
}

operation::ProgramWithCallbacks ArgMax::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    const auto normalized_dim = dim.has_value() ? *dim + input_tensor.get_padded_shape().rank() * (*dim < 0) : dim;
    if (use_multicore) {
        return detail::argmax_multi_core(input_tensor, output_tensor, normalized_dim, sub_core_grids);
    }
    return detail::argmax_single_core(input_tensor, output_tensor, normalized_dim);
}

}  // namespace ttnn::operations::reduction
