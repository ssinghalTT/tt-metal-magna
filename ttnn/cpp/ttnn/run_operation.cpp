// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

#include "ttnn/old_infra_device_operation.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tracy/Tracy.hpp>
#include <tt_stl/reflection.hpp>
#include "tools/profiler/op_profiler.hpp"
#include "ttnn/config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace tt::tt_metal::operation {

namespace detail {

IDevice* get_device(const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors) {
    for (auto& input_tensor : input_tensors) {
        if (std::holds_alternative<DeviceStorage>(input_tensor.tensor_attributes->storage)) {
            return input_tensor.workers.at(0);
        }
    }
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and
            std::holds_alternative<DeviceStorage>(optional_input_tensor.value().tensor_attributes->storage)) {
            return optional_input_tensor.value().workers.at(0);
        }
    }
    auto device = ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to operation are on device");
    return device;
}

template <typename T>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
constexpr bool is_optional_v = is_optional<T>::value;

template <class T>
Tensor* get_tensor(T& maybe_tensor) {
    Tensor* output_tensor = nullptr;
    if constexpr (is_optional_v<T>) {
        if (maybe_tensor.has_value()) {
            output_tensor = &maybe_tensor.value();
        }
    } else {
        output_tensor = &maybe_tensor;
    }
    return output_tensor;
}

void check_output(auto& output_tensors, const std::vector<IDevice*>& workers) {
    for (auto& output_tensor_like : output_tensors) {
        auto output_tensor = get_tensor(output_tensor_like);
        if (!output_tensor) {
            continue;
        }
        TT_FATAL(
            output_tensor->workers.size(),
            "Worker threads must be specified for outputs populated by launch_op. This API can only be used for "
            "creating output tensors on device.");
        TT_FATAL(
            output_tensor->workers == workers,
            "Worker threads must be consistent across all outputs populated by launch_op.");
    }
}

auto& get_workers(auto& output_tensors) {
    for (auto& output_tensor_like : output_tensors) {
        Tensor* output_tensor = get_tensor(output_tensor_like);
        if (output_tensor) {
            return output_tensor->workers;
        }
    }
    TT_THROW("Workers not found in output tensors.");
}

}  // namespace detail
}  // namespace tt::tt_metal::operation

namespace tt::tt_metal::operation {

template <class OutputTensors>
OutputTensors run(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    QueueId cq_id) {
    if constexpr (std::is_same_v<OutputTensors, Tensors>) {
        return ttnn::prim::old_infra_device_operation(
            cq_id, std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors);
    } else {
        return ttnn::prim::old_infra_device_operation_with_optional_output_tensors(
            cq_id, std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors);
    }
}

template Tensors run(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    QueueId cq_id);

template OptionalTensors run(
    DeviceOperation<OptionalTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    QueueId cq_id);

template <class OutputTensors>
OutputTensors run_without_autoformat(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    QueueId cq_id) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    ZoneScoped;
    IDevice* device = detail::get_device(input_tensors, optional_input_tensors);
    Tensors input_tensors_on_dev;
    input_tensors_on_dev.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(input_tensor, device));
        } else {
            input_tensors_on_dev.push_back(input_tensor);
        }
    }
    OptionalConstTensors optional_input_tensors_on_dev;
    optional_input_tensors_on_dev.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() != StorageType::DEVICE) {
            optional_input_tensors_on_dev.push_back(
                AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run<OutputTensors>(
        std::move(operation), input_tensors_on_dev, optional_input_tensors_on_dev, optional_output_tensors, cq_id);
}

template Tensors run_without_autoformat<Tensors>(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    QueueId cq_id);

template OptionalTensors run_without_autoformat<OptionalTensors>(
    DeviceOperation<OptionalTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    QueueId cq_id);

std::vector<Shape> extract_padded_shapes(
    const std::vector<ttnn::TensorSpec>& tensor_specs,
    const std::function<TensorLayout(size_t idx)>& layout_provider,
    const bool use_tensor_layout_from_tensor_spec) {
    std::vector<Shape> padded_shapes;
    padded_shapes.reserve(tensor_specs.size());
    for (size_t idx = 0; idx < tensor_specs.size(); idx++) {
        const auto& tensor_spec = tensor_specs[idx];
        TensorLayout tensor_layout =
            use_tensor_layout_from_tensor_spec ? tensor_spec.tensor_layout() : layout_provider(idx);
        auto logical_shape = tensor_spec.logical_shape();
        padded_shapes.push_back(tensor_layout.compute_padded_shape(logical_shape));
    }
    return padded_shapes;
}

// To be deprecated/removed in favor of new implementation where ops specifically request how to format inputs/outputss
Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    const float pad_value,
    QueueId cq_id) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    ZoneScoped;
    IDevice* device = detail::get_device(input_tensors, optional_input_tensors);

    Tensors formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_padded_shape());
        auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
        if (pad_input) {
            formatted_input_tensors.push_back(
                AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
        } else {
            formatted_input_tensors.push_back(input_tensor);
        }
    }

    OptionalConstTensors formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value()) {
            auto& input_tensor = optional_input_tensor.value();
            auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_padded_shape());
            auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
            if (pad_input) {
                formatted_optional_input_tensors.push_back(
                    AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
            } else {
                formatted_optional_input_tensors.push_back(input_tensor);
            }
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensor);
        }
    }

    auto output_specs = operation.compute_output_specs(input_tensors, optional_output_tensors);
    auto output_tensors = run<Tensors>(
        std::move(operation),
        formatted_input_tensors,
        formatted_optional_input_tensors,
        optional_output_tensors,
        cq_id);

    auto padded_output_shapes = extract_padded_shapes(
        std::move(output_specs),
        [&](size_t idx) {
            auto tensor = output_tensors[idx];
            return TensorLayout(tensor.get_dtype(), Layout::TILE, tensor.memory_config());
        },
        /*use_tensor_layout_from_tensor_spec=*/true);

    TT_ASSERT(output_tensors.size() == padded_output_shapes.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] =
            AutoFormat::format_output_tensor(output_tensors[i], padded_output_shapes[i], device, Layout::TILE);
    }
    return output_tensors;
}

Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const OptionalConstTensors& optional_input_tensors,
    const std::vector<std::optional<FormatParams>>& optional_input_formatting,
    const OptionalTensors& optional_output_tensors,
    ttnn::QueueId cq_id) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    ZoneScoped;
    IDevice* device = detail::get_device(input_tensors, optional_input_tensors);

    TT_ASSERT(input_tensors.size() == input_formatting.size());
    TT_ASSERT(optional_input_tensors.size() == optional_input_formatting.size());

    Tensors formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        formatted_input_tensors.push_back(AutoFormat::format_input_tensor(
            input_tensors[i],
            device,
            input_formatting[i].pad_shape,
            input_formatting[i].pad_value,
            input_formatting[i].target_layout));
    }

    OptionalConstTensors formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (uint32_t i = 0; i < optional_input_tensors.size(); ++i) {
        if (optional_input_tensors[i].has_value()) {
            auto& input_tensor = optional_input_tensors[i].value();
            TT_ASSERT(optional_input_formatting[i].has_value());
            auto& input_formatting = optional_input_formatting[i].value();
            formatted_optional_input_tensors.push_back(AutoFormat::format_input_tensor(
                input_tensor,
                device,
                input_formatting.pad_shape,
                input_formatting.pad_value,
                input_formatting.target_layout));
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensors[i]);
        }
    }

    auto output_specs = operation.compute_output_specs(input_tensors, optional_output_tensors);
    auto output_tensors = run<Tensors>(
        std::move(operation),
        formatted_input_tensors,
        formatted_optional_input_tensors,
        optional_output_tensors,
        cq_id);

    auto legacy_output_shapes = extract_padded_shapes(
        std::move(output_specs),
        [&](size_t idx) {
            auto tensor = output_tensors[idx];
            return TensorLayout(tensor.get_dtype(), output_layouts[idx], tensor.memory_config());
        },
        /*use_tensor_layout_from_tensor_spec=*/false);

    TT_ASSERT(output_tensors.size() == legacy_output_shapes.size());
    TT_ASSERT(output_tensors.size() == output_layouts.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] =
            AutoFormat::format_output_tensor(output_tensors[i], legacy_output_shapes[i], device, output_layouts[i]);
    }

    return output_tensors;
}

std::vector<IDevice*> get_workers_for_op_output(
    const std::vector<Tensor>& inputs, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    std::vector<IDevice*> workers_for_op = {};
    // Infer output workers from inputs. For multi-device tensors the number
    // of workers used for the op (and assigned to the ouput) is the minimum
    // number of workers across all inputs. Additionally, in this case, at least
    // 1 worker must be specified across all inputs, i.e. host inputs are not allowed.
    size_t min_workers_size = std::numeric_limits<uint32_t>::max();
    for (auto& input : inputs) {
        auto workers = input.get_workers();
        min_workers_size = std::min(min_workers_size, workers.size());
        if (workers.size() == min_workers_size) {
            workers_for_op = workers;
        }
    }

    if (not workers_for_op.size()) {
        for (auto& input : optional_inputs) {
            if (input.has_value()) {
                auto workers = input.value().get_workers();
                min_workers_size = std::min(min_workers_size, workers.size());
                if (workers.size() == min_workers_size) {
                    workers_for_op = workers;
                }
            }
        }
    }
    return workers_for_op;
}

template <class OutputType>
void launch_op_func(
    const std::function<OutputType(const Tensors&, const OptionalConstTensors&, const OptionalTensors&)>& op_func,
    const Tensors input_tensors,
    OutputType& output_tensors,
    const OptionalConstTensors optional_input_tensors,
    const OptionalTensors optional_output_tensors) {
    // Send host side op compile and run to the worker queue
    // Assert to ensure that worker threads are specified.
    ZoneScopedN("LaunchOp");
    output_tensors = op_func(input_tensors, optional_input_tensors, optional_output_tensors);
    return;
}

template void launch_op_func<Tensors>(
    const std::function<Tensors(const Tensors&, const OptionalConstTensors&, const OptionalTensors&)>& op_func,
    const Tensors input_tensors,
    Tensors& output_tensors,
    const OptionalConstTensors optional_input_tensors,
    const OptionalTensors optional_output_tensors);
template void launch_op_func<OptionalTensors>(
    const std::function<OptionalTensors(const Tensors&, const OptionalConstTensors&, const OptionalTensors&)>& op_func,
    const Tensors input_tensors,
    OptionalTensors& output_tensors,
    const OptionalConstTensors optional_input_tensors,
    const OptionalTensors optional_output_tensors);

}  // namespace tt::tt_metal::operation
