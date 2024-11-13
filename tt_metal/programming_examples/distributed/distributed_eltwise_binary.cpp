// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/distributed/distributed.hpp"
#include "common/bfloat16.hpp"

#include <magic_enum.hpp>

using namespace tt;
using namespace distributed;
using namespace tt_metal;

// This Programming example illustrates how a set of data-parallel eltwise binary operations can be scheduled on a Virtual Mesh
// through the TT-Mesh infrastructure.
// It covers:
//  MeshDevice Initialization: Creating multiple Virtual Meshes on a single Physical Cluster
//  Virtual Command Queues
//  MeshBuffer creation
//  MeshWorkload Creation
//  Scheduling Host <--> Mesh Data-Movement and Compute through independent VCQs
//  Synchronizing across VCQs to ensure ordering

std::shared_ptr<MeshWorkload> create_binary_mesh_workload(BufferHandle src_0, BufferHandle src_1, BufferHandle dst, uint32_t single_tile_size, uint32_t num_tiles_per_device, BinaryOpType op_type) {
    // Fully specify a program laid out on a single device, to run an Eltwise Binary Op. The code here is directly copied from tt_metal/programming_examples/etlwise_binary/eltwise_binary.cpp
    Program program = CreateProgram();
    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 2;
    constexpr CoreCoord core = {0, 0};

    CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = CB::c_in1;
    CircularBufferConfig cb_src1_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, single_tile_size);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}}).set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    KernelHandle binary_reader_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::vector<uint32_t> compute_kernel_args = {
    };

    constexpr bool fp32_dest_acc_en = false;
    constexpr bool math_approx_mode = false;

    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = get_defines(op_type)
        }
    );
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {
            GetMeshBuffer(src_0).address(),
            static_cast<uint32_t>(GetMeshBuffer(src_0).noc_coordinates().x),
            static_cast<uint32_t>(GetMeshBuffer(src_0).noc_coordinates().y),
            num_tiles_per_device,
            src_1->address(),
            static_cast<uint32_t>(GetMeshBuffer(src_1).noc_coordinates().x),
            static_cast<uint32_t>(GetMeshBuffer(src_1).->noc_coordinates().y),
            num_tiles_per_device,
            0
        }
    );

    SetRuntimeArgs(
        program,
        eltwise_binary_kernel_id,
        core,
        {
            num_tiles_per_device, 1
        }
    );

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            GetMeshBuffer(dst).address(),
            static_cast<uint32_t>(GetMeshBuffer(dst).noc_coordinates().x),
            static_cast<uint32_t>(GetMeshBuffer(dst).noc_coordinates().y),
            num_tiles_per_device
        }
    );
    std::shared_ptr<MeshWorkload> mesh_workload = std::make_shared<MeshWorkload>(std::move(CreateMeshWorkload()));
    InsertProgramInMeshWorkload(*mesh_workload, program);
    return mesh_workload;
}

int main(int argc, char **argv) {
    // Setup workload parameters: shapes and per-device shard sizes
    constexpr uint32_t VIRTUAL_MESH_ROWS = 8;
    constexpr uint32_t VIRTUAL_MESH_COLS = 4;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t FP16_DATUM_SIZE_BYTES = 2;
    constexpr uint32_t GLOBAL_TENSOR_SHAPE_X = 128;
    constexpr uint32_t GLOBAL_TENSOR_SHAPE_Y = 256;
    // Compute runtime configs based on workload parameters
    constexpr uint32_t single_tile_size = FP16_DATUM_SIZE_BYTES * TILE_HEIGHT * TILE_WIDTH;
    constexpr uint32_t global_tensor_shape = ttnn::SimpleShape({1, 1, GLOBAL_TENSOR_SHAPE_Y, GLOBAL_TENSOR_SHAPE_X});
    constexpr uint32_t device_shard_x = GLOBAL_TENSOR_SHAPE_X / VIRTUAL_MESH_COLS;
    constexpr uint32_t device_shard_y = GLOBAL_TENSOR_SHAPE_Y / VIRTUAL_MESH_ROWS;
    constexpr uint32_t device_shard_shape = ttnn::SimpleShape({1, 1, device_shard_y, device_shard_x});
    constexpr uint32_t num_tiles_per_device = device_shard_x * device_shard_y / (TILE_HEIGHT * TILE_WIDTH);
    constexpr uint32_t dram_buffer_size_per_device = single_tile_size * num_tiles_per_device;
    constexpr uint32_t distributed_buffer_size = GLOBAL_TENSOR_SHAPE_X * GLOBAL_TENSOR_SHAPE_Y * FP16_DATUM_SIZE_BYTES;

    // Create 2 8x4 Virtual Meshes on an 8x8 Physical Mesh
    MeshType mesh_type = MeshType::RowMajor;
    MeshShape virtual_mesh_shape = {VIRTUAL_MESH_ROWS, VIRTUAL_MESH_COLS};

    MeshConfig mesh_config_0 = MeshConfig{.shape = virtual_mesh_shape, .offset = MeshOffset(0, 0), .type = mesh_type};
    MeshConfig mesh_config_1 = MeshConfig{.shape = virtual_mesh_shape, .offset = MeshOffset(0, 4), .type = mesh_type};

    DeviceHandle virtual_mesh_0 = CreateMeshDevice(
                                                mesh_config_0,
                                                2, /* num_cqs */
                                                DEFAULT_L1_SMALL_SIZE,
                                                DEFAULT_TRACE_REGION_SIZE);
    DeviceHandle virtual_mesh_1 = CreateMeshDevice(
                                                mesh_config_1,
                                                2, /* num_cqs */
                                                DEFAULT_L1_SMALL_SIZE,
                                                DEFAULT_TRACE_REGION_SIZE);

    // At this point we have 2 handles to separate virtual meshes
    // Obtain handles for both VCQs across both meshes
    CommandQueueHandle virtual_mesh_0_cq_0_handle = GetCommandQueue(virtual_mesh_0, 0);
    CommandQueueHandle virtual_mesh_0_cq_1_handle = GetCommandQueue(virtual_mesh_0, 1);
    CommandQueueHandle virtual_mesh_1_cq_0_handle = GetCommandQueue(virtual_mesh_1, 0);
    CommandQueueHandle virtual_mesh_1_cq_1_handle = GetCommandQueue(virtual_mesh_1, 1);

    // Create DistributedBuffers that are sharded across devices and DRAM interleaved within the Device Local Address Space
    DeviceLocalLayoutConfig per_device_buffer_config {
                .page_size = dram_buffer_size_per_device,
                .buffer_layout = TensorMemoryLayout::INTERLEAVED,
    };

    // Specify how the DistributedBuffers live inside the memory exposed on both Virtual Mesh
    ShardedBufferConfig distributed_buffer_config_virtual_mesh_0 {
        .mesh_device = virtual_mesh_0;
        .buffer_type = BufferType::DRAM,
        .global_tensor_shape = global_tensor_shape,
        .distributed_shard_shape = device_shard_shape,
        .global_buffer_size = distributed_buffer_size,
        .device_shard_layout = per_device_buffer_config
    };

    ShardedBufferConfig distributed_buffer_config_virtual_mesh_1 {
        .mesh_device = virtual_mesh_1;
        .buffer_type = BufferType::DRAM,
        .global_tensor_shape = global_tensor_shape,
        .distributed_shard_shape = device_shard_shape,
        .global_buffer_size = distributed_buffer_size,
        .device_shard_layout = per_device_buffer_config
    };

    // Allocate Buffers used for IO in the workload
    //  ======== These Buffers live on Virtual Mesh 0 ========
    BufferHandle mul_src_0 = CreateDistributedBuffer(distributed_buffer_config_virtual_mesh_0);
    BufferHandle mul_src_1 = CreateDistributedBuffer(distributed_buffer_config_virtual_mesh_0);
    BufferHandle mul_dst = CreateDistributedBuffer(distributed_buffer_config_virtual_mesh_0);
    //  ======== These Buffers live on Virtual Mesh 1 ========
    BufferHandle add_src_0 = CreateDistributedBuffer(distributed_buffer_config_virtual_mesh_1);
    BufferHandle add_src_1 = CreateDistributedBuffer(distributed_buffer_config_virtual_mesh_1);
    BufferHandle add_dst = CreateDistributedBuffer(distributed_buffer_config_virtual_mesh_1);

    // Create MeshWorkloads for both Virtual Meshes. Each runs an independent binary operation (Mul on Mesh 0 and Add on Mesh 1)
    std::shared_ptr<MeshWorkload> mul_mesh_workload = create_binary_mesh_workload(mul_src_0, mul_src_1, mul_dst, single_tile_size, num_tiles_per_device, BinaryOpType::MUL);
    std::shared_ptr<MeshWorkload> add_mesh_workload = create_binary_mesh_workload(add_src_0, add_src_1, add_dst, single_tile_size, num_tiles_per_device, BInaryOpType::ADD);

    // Create randomized input data to drive the workload
    std::vector<uint32_t> random_data_0 = create_random_vector_of_bfloat16(distributed_buffer_size, 1, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> random_data_1 = create_random_vector_of_bfloat16(distributed_buffer_size, 1, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> random_data_2 = create_random_vector_of_bfloat16(distributed_buffer_size, 1, std::chrono::system_clock::now().time_since_epoch().count());

    // Setup output containers we will read data from the Virtual Meshes into
    std::vector<uint32_t> mul_readback_data = {};
    std::vector<uint32_t> add_readback_data = {};

    // Data-Movement and Compute on Virtual Mesh 0. IO on CQ1, compute on CQ0. Use events to ensure ordering.
    std::shared_ptr<MeshEvent> virtual_mesh_0_write_event = std::make_shared<MeshEvent>();
    std::shared_ptr<MeshEvent> virtual_mesh_1_compute_event = std::make_shared<MeshEvent>();

    // Write inputs
    EnqueueWriteBuffer(virtual_mesh_0_cq_1_handle, mul_src_0, random_data_0);
    EnqueueWriteBuffer(virtual_mesh_0_cq_1_handle, mul_src_1, random_data_1);
    // Record that inputs were written
    EnqueueRecordMeshEvent(virtual_mesh_0_cq_1_handle, virtual_mesh_0_write_event);
    // Wait until inputs were written
    EnqueueWaitForMeshEvent(virtual_mesh_0_cq_0_handle, virtual_mesh_0_write_event);
    // Run compute
    EnqueueMeshWorkload(virtual_mesh_0_cq_0_handle, *mul_mesh_workload);
    // Record that compute was run and is completed
    EnqueueRecordMeshEvent(virtual_mesh_0_cq_0_handle, virtual_mesh_1_compute_event);
    // Wait until compute has completed
    EnqueueWaitForMeshEvent(virtual_mesh_0_cq_1_handle, virtual_mesh_1_compute_event);
    // Read outputs
    EnqueueReadBuffer(virtual_mesh_0_cq_1_handle, mul_dst, mul_readback_data);

    // Data-Movement and Compute on Virtual Mesh 1. IO and compute on CQ0. No need to use events to synchronize.
    // Write inputs
    EnqueueWriteBuffer(virtual_mesh_1_cq_0_handle, add_src_0, mul_readback_data);
    EnqueueWriteBuffer(virtual_mesh_1_cq_0_handle, add_src_1, random_data_2);
    // Run compute
    EnqueueMeshWorkload(virtual_mesh_1_cq_0_handle, *add_mesh_workload);
    // Read outputs
    EnqueueReadBuffer(virtual_mesh_1_cq_0_handle, add_dst, add_readback_data);

    CloseDevice(virtual_mesh_0);
    CloseDevice(virtual_mesh_1);
}
