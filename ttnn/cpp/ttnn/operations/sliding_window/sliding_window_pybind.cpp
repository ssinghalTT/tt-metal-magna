// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/pybind11/decorators.hpp"

#include "sliding_window.hpp"
#include "halo/halo.hpp"

using namespace tt::tt_metal;

namespace py = pybind11;
namespace ttnn::operations::sliding_window::detail {

void bind_parallel_config(py::module& module) {
    py::class_<ParallelConfig>(module, "ParallelConfig")
        .def(
            py::init<CoreRangeSet, TensorMemoryLayout, ShardOrientation>(),
            py::kw_only(),
            py::arg("grid"),
            py::arg("shard_scheme"),
            py::arg("shard_orientation"))
        .def_readwrite("grid", &ParallelConfig::grid)
        .def_readwrite("shard_scheme", &ParallelConfig::shard_scheme)
        .def_readwrite("shard_orientation", &ParallelConfig::shard_orientation);
}

void bind_halo(py::module& module) {
    const auto doc = R"doc(
            Halo exchange operations
        )doc";

    using OperationType = decltype(ttnn::halo);

    ttnn::bind_registered_operation(
        module,
        ttnn::halo,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input_tensor,
               const SlidingWindowConfig& config,
               uint32_t pad_val,
               bool remote_read,
               bool transpose_mcast,
               const tt::tt_metal::MemoryConfig& output_memory_config,
               bool is_out_tiled,
               bool in_place,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    config,
                    pad_val,
                    remote_read,
                    transpose_mcast,
                    output_memory_config,
                    is_out_tiled,
                    in_place);
            },
            py::arg("input_tensor"),
            py::arg("config"),
            py::kw_only(),
            py::arg("pad_val") = 0,
            py::arg("remote_read") = false,
            py::arg("transpose_mcast") = true,
            py::arg("output_memory_config") = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("is_out_tiled") = true,
            py::arg("in_place") = false,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::sliding_window::detail
