// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scatter_enums.hpp"

#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::experimental {

struct ScatterOperation {
    static Tensor invoke(
        const int32_t& dim,
        const Tensor& input_tensor,
        const Tensor& index_tensor,
        const std::optional<scatter::ScatterReductionType>& opt_reduction,
        const std::optional<MemoryConfig>& opt_out_memory_config,
        std::optional<Tensor> opt_output);

    static Tensor preprocess_input(const Tensor& input_tensor, const int32_t& di);
    static Tensor postprocess_output();
};

}  // namespace operations::experimental

namespace experimental {
constexpr auto scatter_ =
    ttnn::register_operation<"ttnn::scatter_", ttnn::operations::experimental::ScatterOperation>();
}  // namespace experimental

}  // namespace ttnn
