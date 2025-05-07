// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter__device_operation.hpp"

#include "scatter__device_operation_types.hpp"
#include "ttnn/operations/experimental/scatter_/device/scatter__program_factory.hpp"

namespace ttnn::operations::experimental::scatter {

Scatter_DeviceOperation::program_factory_t Scatter_DeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return Scatter_ProgramFactory{};
}

void Scatter_DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void Scatter_DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const int32_t& dim{args.dim};
    const auto& input_tensor{tensor_args.input_tensor};
    const auto& index_tensor{tensor_args.index_tensor};
    const auto& input_shape{input_tensor.get_padded_shape()};
    const auto& index_shape{index_tensor.get_padded_shape()};
    const uint32_t input_rank{input_shape.rank()};
    const uint32_t index_rank{index_shape.rank()};

    if (tensor_args.opt_output.has_value()) {
        const auto& output_tensor{tensor_args.opt_output.value()};
        const auto& output_shape{output_tensor.get_padded_shape()};
        const auto& output_rank{output_shape.rank()};

        TT_FATAL(
            output_shape == index_shape,
            "The shape of the output tensor should be the same as the input index tensor's shape. (output_shape: {}, "
            "input_shape: {})",
            output_shape,
            index_shape);

        TT_FATAL(
            dim >= -output_rank,
            "dim cannot be lower than output shape's negative rank (dim: {}, rank: {}).",
            dim,
            -input_rank);
        TT_FATAL(
            dim < output_rank,
            "dim must be lower than output shape's positive rank (dim: {}, rank: {}).",
            dim,
            input_rank);
    }

    for (uint32_t probe_dim = 0; probe_dim < input_shape.rank(); ++probe_dim) {
        TT_FATAL(
            index_shape[probe_dim] <= input_shape[probe_dim],
            "Index tensor has a bigger dimension no {} than input shape (index dimension: {}, input_dimension: "
            "{}).",
            probe_dim,
            index_shape[probe_dim],
            input_shape[probe_dim]);
    }

    TT_FATAL(
        dim >= -input_rank,
        "dim cannot be lower than input shape's negative rank (dim: {}, rank: {}).",
        dim,
        -input_rank);
    TT_FATAL(
        dim < input_rank, "dim must be lower than input shape's positive rank (dim: {}, rank: {}).", dim, input_rank);

    TT_FATAL(!input_tensor.is_sharded(), "Sharded tensors are not supported - input_tensor is sharded.");
    TT_FATAL(!index_tensor.is_sharded(), "Sharded tensors are not supported - index_tensor is sharded.");

    TT_FATAL(
        input_tensor.get_layout() == Layout::TILE,
        "Input tensor doesn't have a tile layout - only tile layout is supported.");
    TT_FATAL(
        index_tensor.get_layout() == Layout::TILE,
        "Index tensor doesn't have a tile layout - only tile layout is supported.");

    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor's buffer is null.");
    TT_FATAL(index_tensor.buffer() != nullptr, "Index tensor's buffer is null.");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be allocated on a device.");
    TT_FATAL(index_tensor.storage_type() == StorageType::DEVICE, "Index tensor must be allocated on a device.");
}

Scatter_DeviceOperation::spec_return_value_t Scatter_DeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    if (tensor_args.opt_output.has_value()) {
        return tensor_args.opt_output.value().get_tensor_spec();
    }

    return TensorSpec{
        tensor_args.index_tensor.get_logical_shape(),
        TensorLayout{tensor_args.input_tensor.get_dtype(), PageConfig(Layout::TILE), args.output_mem_config}};
}

Scatter_DeviceOperation::tensor_return_value_t Scatter_DeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.opt_output.has_value()) {
        return *tensor_args.opt_output;
    }

    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

Scatter_DeviceOperation::invocation_result_t Scatter_DeviceOperation::invoke(
    const int32_t& dim,
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const MemoryConfig& output_memory_config,
    const std::optional<ScatterReductionType>& opt_reduction,
    std::optional<Tensor>& opt_output) {
    return {
        operation_attributes_t{dim, output_memory_config, opt_reduction},
        tensor_args_t{input_tensor, index_tensor, opt_output}};
}

}  // namespace ttnn::operations::experimental::scatter
