// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <variant>

// #include "core_coord.h"
#include "common/core_coord.h"
#include "impl/kernels/kernel_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::generic {

struct circular_buffer_attributes_t {
    CoreRangeSet core_spec = {{}};
    uint32_t total_size;
    uint32_t page_size;
    // uint8_t buffer_index;
    tt::DataFormat data_format;

    // this needs better solution as we now have input tensors (std::vector) and output tensor so index is not great
    std::optional<int> set_globally_allocated_address = std::nullopt; // an index to io_tensors that will set globally allocated address on CB
};

struct data_movement_attributes_t {
    CoreRangeSet core_spec= {{}};
    std::string kernel_path;
    tt::tt_metal::DataMovementConfig config;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> runtime_args_per_core = {};

    // std::variant<CoreCoord, CoreRange, CoreRangeSet> core_spec;
    // std::shared_ptr<RuntimeArgs> runtime_args;
    // std::vector<std::shared_ptr<RuntimeArgs>> runtime_args;

};

struct compute_attributes_t {
    CoreRangeSet core_spec = {{}};
    std::string  kernel_path;
    tt::tt_metal::ComputeConfig config;
    // std::vector<uint32_t> runtime_args = {};
    std::unordered_map<CoreCoord, std::vector<uint32_t>> runtime_args_per_core = {};

};

struct GenericOpDeviceOperation {

    struct operation_attributes_t {

        std::unordered_map<uint8_t, circular_buffer_attributes_t> circular_buffer_attributes;

        std::vector<data_movement_attributes_t> data_movement_attributes;

        // Ethernet not supported at the moment.
        // std::optional<tt::metal::EthernetConfig> ethernetConfig;

        std::vector<compute_attributes_t> compute_attributes;
    };


    // Define the return types for the shape(s) of the operation
    using shape_return_value_t = ttnn::Shape;

    // Define the return types for the tensor(s) of the operation
    // Can be a single Tensor, std::optional<Tensor, ...>, std::vector<Tensor>, std::tuple<Tensor, ...> etc.
    using tensor_return_value_t = Tensor;

    struct tensor_args_t {
        // std::vector<std::reference_wrapper<Tensor>> io_tensors;
        std::vector<Tensor> io_tensors;
        // std::vector<Tensor> input_tensors; // this might not be the best thing?
        // const std::vector<Tensor>& input_tensors;
        // const Tensor& input_tensor;
        // Tensor& output_tensor;
        // reflections assume there are no two params of the same type?
        // note: in instantiation of function template specialization 'reflect::size<ttnn::operations::generic::GenericOpDeviceOperation::tensor_args_t>' requested here
    };

    // Program factories
    struct GenericProgram {
        // to refactor this when we implement caching
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // where/when is this used?
        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<GenericProgram>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

};  // struct GenericOpDeviceOperation

}   // namespace ttnn::operations::generic
