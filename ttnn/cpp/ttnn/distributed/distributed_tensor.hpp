// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::distributed {

// Mapper interface that distributes a host tensor onto a multi-device configuration.
class TensorToMesh {
public:
    virtual ~TensorToMesh() = default;
    virtual std::vector<Tensor> map(const Tensor& tensor) const = 0;
    virtual tt::tt_metal::DistributedTensorConfig config() const = 0;
};

// Composer interface that aggregates a multi-device tensor into a host tensor.
class MeshToTensor {
public:
    virtual ~MeshToTensor() = default;
    virtual Tensor compose(const std::vector<Tensor>& tensors) const = 0;
};

// Creates a mapper that replicates a tensor across all devices.
std::unique_ptr<TensorToMesh> replicate_tensor_to_mesh_mapper(MeshDevice& mesh_device);

// Creates a mapper that shards a tensor along a single dimension.
std::unique_ptr<TensorToMesh> shard_tensor_to_mesh_mapper(MeshDevice& mesh_device, int dim);

// Creates a mapper that shards a tensor along two dimensions, which will be intepreted as rows and columns.
// If either dimension is not specified, the tensor is replicated along that dimension.
struct Shard2dConfig {
    std::optional<int> row_dim;
    std::optional<int> col_dim;
};
std::unique_ptr<TensorToMesh> shard_tensor_to_2d_mesh_mapper(
    MeshDevice& mesh_device, const MeshShape& mesh_shape, const Shard2dConfig& config);

// Creates a composer that concatenates a tensor across a single dimension.
std::unique_ptr<MeshToTensor> concat_mesh_to_tensor_composer(int dim);

// Creates a composer that concatenates a tensor across two dimensions.
struct Concat2dConfig {
    int row_dim = -1;
    int col_dim = -1;
};
std::unique_ptr<MeshToTensor> concat_2d_mesh_to_tensor_composer(MeshDevice& mesh_device, const Concat2dConfig& config);

// TODO: #20895 - ND `create_mesh_mapper` and `create_mesh_composer` are generalized ND interfaces that will supercede
// all existing mapper and composer types.

struct MeshMapperConfig {
    // Specifies the tensor should be replicated across devices.
    struct Replicate {};

    // Specifies the tensor should be sharded along the specified dimension.
    struct Shard {
        int dim = 0;
    };

    // Specifies placements for each dimension of the shape.
    // The size of `placements` must match the dimensions of the shape.
    //
    // For example, sharding a 2x8 tensor over 2x2 mesh with {Replicate(), Shard{1}} will yield the following result:
    //
    //    Input Tensor [2, 8]:
    // +----+----+----+----+----+----+---+-----+
    // |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
    // |----+----+----+----+----+----+---+-----+
    // |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
    // +----+----+----+----+----+----+---+-----+
    //
    //    Shape [2, 2]:
    // +-------+-------+
    // | (0,0) | (0,1) |
    // +-------+-------+
    // | (1,0) | (1,1) |
    // +-------+-------+
    //
    // Distributed Tensor on Mesh (placements = {Replicate{}, Shard{1}}):
    //
    // +-------------------------------------+---------------------------------------+
    // |  (0,0)                              |  (0,1)                                |
    // | +---+---+---+---+---+---+----+----+ | +---+---+---+---+----+----+----+----+ |
    // | | 0 | 1 | 2 | 3 | 8 | 9 | 10 | 11 | | | 4 | 5 | 6 | 7 | 12 | 13 | 14 | 15 | |
    // | +---+---+---+---+---+---+----+----+ | +---+---+---+---+----+----+----+----+ |
    // +-------------------------------------+---------------------------------------+
    // |  (1,0)                              |  (1,1)                                |
    // | +---+---+---+---+---+---+----+----+ | +---+---+---+---+----+----+----+----+ |
    // | | 0 | 1 | 2 | 3 | 8 | 9 | 10 | 11 | | | 4 | 5 | 6 | 7 | 12 | 13 | 14 | 15 | |
    // | +---+---+---+---+---+---+----+----+ | +---+---+---+---+----+----+----+----+ |
    // +-------------------------------------+---------------------------------------+
    //
    std::vector<std::variant<Replicate, Shard>> placements;
};

// Creates an ND mesh mapper that distributes a tensor according to the `config`.
// If `shape` is not provided, the shape of `mesh_device` is used.
// Otherwise, the size of the shape must match the size of the mesh device shape.
std::unique_ptr<TensorToMesh> create_mesh_mapper(
    MeshDevice& mesh_device,
    const MeshMapperConfig& config,
    const std::optional<ttnn::MeshShape>& shape = std::nullopt);

struct MeshComposerConfig {
    // Specifies dimension of the tensor to concatenate.
    std::vector<int> dims;
};

// Creates an ND mesh composer that aggregates a tensor according to the `config`.
// If `shape` is not provided, the shape of `mesh_device` is used.
// Otherwise, the size of the shape must match the size of the mesh device shape.
std::unique_ptr<MeshToTensor> create_mesh_composer(
    MeshDevice& mesh_device,
    const MeshComposerConfig& config,
    const std::optional<ttnn::MeshShape>& shape = std::nullopt);

// Distributes a host tensor onto multi-device configuration according to the `mapper`.
Tensor distribute_tensor(
    const Tensor& tensor,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device = std::nullopt);

// Aggregates a multi-device tensor into a host tensor according to the `composer`.
Tensor aggregate_tensor(const Tensor& tensor, const MeshToTensor& composer);

}  // namespace ttnn::distributed
