// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>

#include "../scatter_enums.hpp"
#include "scatter__device_operation_types.hpp"
#include "ttnn/operations/experimental/scatter_/scatter_.hpp"

#include "scatter__program_factory.hpp"

namespace ttnn::operations::experimental::scatter {

struct Scatter_DeviceOperation {
    using operation_attributes_t = scatter::operation_attributes_t;
    using tensor_args_t = scatter::tensor_args_t;
    using spec_return_value_t = scatter::spec_return_value_t;
    using tensor_return_value_t = scatter::tensor_return_value_t;
    using program_factory_t = std::variant<scatter::Scatter_ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;

    static invocation_result_t invoke(
        const int32_t& dim,
        const Tensor& input_tensor,
        const Tensor& index_tensor,
        const MemoryConfig& output_memory_config,
        const std::optional<ScatterReductionType>& opt_reduction,
        std::optional<Tensor>& opt_output);
};

}  // namespace ttnn::operations::experimental::scatter

namespace ttnn::prim {
constexpr auto scatter_ = ttnn::
    register_operation<"ttnn::prim::scatter_", ttnn::operations::experimental::scatter::Scatter_DeviceOperation>();
}
