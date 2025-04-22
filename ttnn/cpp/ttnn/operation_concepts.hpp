// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <optional>
#include <random>
#include <type_traits>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/program_cache.hpp>

#include <tt_stl/reflection.hpp>

#include "ttnn/distributed/types.hpp"

namespace ttnn::device_operation {

template <typename T>
concept ProgramFactoryConcept = requires {
    typename T::cached_program_t;

    [](const auto& operation_attributes, const auto& tensor_args, auto& tensor_return_value) {
        auto cached_program = T::create(operation_attributes, tensor_args, tensor_return_value);

        T::override_runtime_arguments(cached_program, operation_attributes, tensor_args, tensor_return_value);
    };
};

template <typename T>
concept MeshWorkloadFactoryConcept = requires {
    typename T::cached_mesh_workload_t;

    [](const auto& operation_attributes,
       const ttnn::MeshCoordinateRangeSet& tensor_coords,
       const auto& tensor_args,
       auto& tensor_return_value) {
        auto cached_workload =
            T::create_mesh_workload(operation_attributes, tensor_coords, tensor_args, tensor_return_value);

        T::override_runtime_arguments(cached_workload, operation_attributes, tensor_args, tensor_return_value);
    };
};

template <typename device_operation_t>
concept HasComputeOutputSpecs = requires(
    device_operation_t op,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    {
        op.compute_output_specs(operation_attributes, tensor_args)
    } -> std::same_as<typename device_operation_t::spec_return_value_t>;
};

template <typename device_operation_t>
concept DeviceOperationWithDescriptorConcept = requires(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    typename device_operation_t::tensor_return_value_t& tensor_return_value) {
    { device_operation_t::validate(operation_attributes, tensor_args) };
    {
        device_operation_t::create_output_tensors(operation_attributes, tensor_args)
    } -> std::same_as<typename device_operation_t::tensor_return_value_t>;
    {
        device_operation_t::create_program(operation_attributes, tensor_args, tensor_return_value)
    } -> std::same_as<tt::tt_metal::ProgramDescriptor>;
};

template <typename device_operation_t>
concept DeviceOperationWithoutDescriptorConcept = requires(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    { device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args) };
    { device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args) };
    {
        device_operation_t::create_output_tensors(operation_attributes, tensor_args)
    } -> std::same_as<typename device_operation_t::tensor_return_value_t>;
    [](const typename device_operation_t::operation_attributes_t& operation_attributes,
       const typename device_operation_t::tensor_args_t& tensor_args) {
        // All program factories returned by `select_program_factory` must implement exactly one of
        // `ProgramFactoryConcept` or `MeshWorkloadFactoryConcept`.
        const auto program_factory = device_operation_t::select_program_factory(operation_attributes, tensor_args);
        std::visit(
            []<typename T>(const T&) { static_assert(ProgramFactoryConcept<T> != MeshWorkloadFactoryConcept<T>); },
            program_factory);
    };
};

template <typename device_operation_t>
concept DeviceOperationConcept = (DeviceOperationWithDescriptorConcept<device_operation_t> ||
                                  DeviceOperationWithoutDescriptorConcept<device_operation_t>) &&
                                 HasComputeOutputSpecs<device_operation_t>;

template <typename device_operation_t>
concept DeviceOperationWithCustomProgramCacheConcept =
    DeviceOperationConcept<device_operation_t> &&
    requires(
        const typename device_operation_t::operation_attributes_t& operation_attributes,
        const typename device_operation_t::tensor_args_t& tensor_args) {
        {
            device_operation_t::compute_program_hash(operation_attributes, tensor_args)
        } -> std::convertible_to<tt::stl::hash::hash_t>;
    };

template <typename device_operation_t>
concept HasSkipLaunch = requires(
    device_operation_t op,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    const typename device_operation_t::tensor_return_value_t& tensor_return_value) {
    {
        device_operation_t::skip_launch(operation_attributes, tensor_args, tensor_return_value)
    } -> std::convertible_to<bool>;
};

}  // namespace ttnn::device_operation
