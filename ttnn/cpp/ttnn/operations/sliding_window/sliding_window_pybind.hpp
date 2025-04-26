// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::sliding_window {

namespace detail {
void bind_parallel_config(pybind11::module& module);
void bind_halo(pybind11::module& module);
}  // namespace detail

void bind_sliding_window(pybind11::module& module) {
    detail::bind_parallel_config(module);
    detail::bind_halo(module);
}

}  // namespace ttnn::operations::sliding_window
