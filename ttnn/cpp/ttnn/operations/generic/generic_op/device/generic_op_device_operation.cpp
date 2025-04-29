// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_device_operation.hpp"
#include <iostream>

namespace ttnn::operations::generic {

GenericOpDeviceOperation::program_factory_t GenericOpDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    return GenericProgram{};
}

void GenericOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void GenericOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

// will different tensors have different specs????
GenericOpDeviceOperation::spec_return_value_t GenericOpDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // User has to do this. Just referencing last element (preallocated output tensor).
    return tensor_args.io_tensors[0].get_tensor_spec();
}

GenericOpDeviceOperation::tensor_return_value_t GenericOpDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Don't create anything, user is passing output tensor.
    // TODO: CHECK IF THEY PASSED AN OUTPUT TENSOR
    return tensor_args.io_tensors.back();

    // return create_device_tensor(
    //     compute_output_specs(operation_attributes, tensor_args), tensor_args.output_tensor.device());
}

std::tuple<GenericOpDeviceOperation::operation_attributes_t, GenericOpDeviceOperation::tensor_args_t> 
GenericOpDeviceOperation::invoke(const std::vector<Tensor>& io_tensors, const operation_attributes_t& operation_attributes) {
    // const std::vector<Tensor> input_tensors(io_tensors.begin(), io_tensors.end() - 1);
    // const Tensor output_tensor = io_tensors.back();
    if (io_tensors.empty()) {
        throw std::invalid_argument("io_tensors must contain at least one tensor.");
    }
    if (io_tensors.size() < 2) {
        throw std::invalid_argument("io_tensors must contain at least one input tensor and one output tensor.");
    }
    return {operation_attributes, tensor_args_t{.io_tensors = io_tensors}};
}

}  // namespace ttnn::operations::generic
