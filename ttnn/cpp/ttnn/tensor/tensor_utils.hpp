// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "types.hpp"

namespace tt {

namespace tt_metal {
ttnn::Shape infer_dims_for_reshape(const Tensor& tensor, tt::stl::Span<const int32_t> shape);

static int compute_flat_indices(tt::stl::Span<const int> indices, tt::stl::Span<const uint32_t> strides) {
    int flat_index = 0;
    for (auto i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
};

static std::size_t compute_buffer_size(const ttnn::Shape& shape, DataType data_type, const Tile& tile) {
    const size_t volume = shape.volume();
    auto tile_hw = tile.get_tile_hw();
    if (data_type == DataType::BFLOAT8_B) {
        auto tile_size_bytes = tile.get_tile_size(DataFormat::Bfp8_b);
        TT_ASSERT(volume % tile_hw == 0);
        const auto bfloat8_b_volume = volume / tile_hw * tile_size_bytes;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat8_b_volume / sizeof(std::uint32_t);
    }
    if (data_type == DataType::BFLOAT4_B) {
        auto tile_size_bytes = tile.get_tile_size(DataFormat::Bfp4_b);
        TT_ASSERT(volume % tile_hw == 0);
        const auto bfloat4_b_volume = volume / tile_hw * tile_size_bytes;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat4_b_volume / sizeof(std::uint32_t);
    }
    return volume;
}

constexpr auto compute_flat_input_index = [](const auto& indices, const auto& strides) {
    uint32_t flat_index = 0;
    for (auto i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
};

bool is_arch_gs(const tt::ARCH& arch);
bool is_arch_whb0(const tt::ARCH& arch);

bool is_cpu_tensor(const Tensor& tensor);
bool is_device_tensor(const Tensor& tensor);

// Given a multi-device tensor, and a function that transforms a tensor, applies the function to all per-device
// tensors.
Tensor transform(const Tensor& tensor, std::function<Tensor(const Tensor&)> transform_func);

// Given a multi-device tensor, and a callable, applies the function to all per-device tensors.
void apply(const Tensor& tensor, const std::function<void(const Tensor&)>& callable);

// Given a multi-device tensor, returns all the devices it is mapped to.
std::vector<IDevice*> get_devices(const Tensor& multi_device_tensor);

uint32_t num_buffers_in_tensor(const Tensor& tensor);

Tensor get_shard_for_device(
    const Tensor& tensor, IDevice* target_device, std::optional<int> buffer_index = std::nullopt);

void insert_buffer_and_shape_for_device(
    IDevice* target_device,
    const Tensor& shard,
    Tensor& tensor_to_modify,
    std::optional<int> buffer_index = std::nullopt);

Tensor copy_borrowed_tensor_in_async_mode(IDevice* worker, const Tensor& tensor);

inline bool is_tensor_on_device(const ttnn::Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

inline bool is_tensor_on_device_or_multidevice(const ttnn::Tensor& tensor) {
    return tensor.storage_type() == StorageType::DEVICE || tensor.storage_type() == StorageType::MULTI_DEVICE;
}

template <class T>
inline uint32_t get_batch_size(const T& shape) {
    uint32_t result = 1;
    for (auto i = 0; i < shape.rank() - 2; i++) {
        result *= shape[i];
    }
    return result;
}

// Useful information about how a shard_shape cuts a 2D shape
// - num_shards_height: Number of shards along the height (including partial last shard, if any)
// - last_shard_height: Height of last partial shard (if None, it will be same as full shard shape height)
// - num_shards_width: Number of shards along the width (including partial last shard, if any)
// - last_shard_width: Width of last partial shard (if None, it will be same as full shard shape width)
struct ShardDivisionSpec {
    size_t num_shards_height = 0;
    size_t last_shard_height = 0;
    size_t num_shards_width = 0;
    size_t last_shard_width = 0;
};

// Returns ShardDivisionSpecs given 2D shape and shard_shape
ShardDivisionSpec compute_shard_division_spec(const Shape2D& shape, const Shape2D& shard_shape);

}  // namespace tt_metal
}  // namespace tt
