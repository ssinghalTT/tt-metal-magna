// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_.hpp"

namespace ttnn {

namespace operations::experimental {

// TODO(jbbieniekTT): output_memory_config?
Tensor ScatterOperation::invoke(
    const int32_t& dim,
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const std::optional<scatter::ScatterReductionType>& opt_reduction,
    const std::optional<MemoryConfig>& output_memory_config,
    std::optional<Tensor> opt_output) {
    //
    Tensor output_tensor;

    return output_tensor;
}

}  // namespace operations::experimental

}  // namespace ttnn
