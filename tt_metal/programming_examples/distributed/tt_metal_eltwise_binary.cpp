// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

#include <magic_enum.hpp>

using namespace tt;
using namespace tt_metal;


// This Programming example illustrates how a set of eltwise binary operations can be scheduled on 2 TT Accelerators
// through the TT-Metal infrastructure.
// It covers:
//  Device Initialization: Creating multiple device handles on a single Physical Cluster
//  Command Queues
//  Buffer creation
//  Program Creation
//  Scheduling Host <--> Device Data-Movement and Compute through independent Command Queues
//  Synchronizing across Command Queues to ensure ordering

std::shared_ptr<Program> create_binary_program(BufferHandle src_0, BufferHandle src_1, BufferHandle dst, uint32_t single_tile_size, uint32_t num_tiles_per_device, BinaryOpType op_type) {
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
    return std::make_shared<Program>(std::move(program));
}

int main(int argc, char **argv) {
    // Setup workload parameters: shapes and per-device shard sizes
    constexpr uint32_t NUM_DEVICES_X = 1;
    constexpr uint32_t NUM_DEVICES_Y = 2;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t FP16_DATUM_SIZE_BYTES = 2;
    constexpr uint32_t TENSOR_SHAPE_X = 128;
    constexpr uint32_t TENSOR_SHAPE_Y = 256;
    // Compute runtime configs based on workload parameters
    constexpr uint32_t single_tile_size = FP16_DATUM_SIZE_BYTES * TILE_HEIGHT * TILE_WIDTH;
    constexpr uint32_t num_tiles_per_device = TENSOR_SHAPE_X * TENSOR_SHAPE_Y / (TILE_HEIGHT * TILE_WIDTH);
    constexpr uint32_t dram_buffer_size_per_device = single_tile_size * num_tiles_per_device;

    // Create 2 Device handles on a 2x1 Physical Mesh
    DeviceHandle device_0 = CreateDevice(
                                0, /* device_id */
                                2, /* num_hw_cqs */
                                DEFAULT_L1_SMALL_SIZE,
                                DEFAULT_TRACE_REGION_SIZE);
    DeviceHandle device_1 = CreateDevice(
                                1, /* device_id */
                                2, /* num_hw_cqs */
                                DEFAULT_L1_SMALL_SIZE,
                                DEFAULT_TRACE_REGION_SIZE);

    // Obtain handles for both command queues across both devices
    CommandQueueHandle device_0_cq_0_handle = GetCommandQueue(device_0, 0);
    CommandQueueHandle device_0_cq_1_handle = GetCommandQueue(device_0, 0);
    CommandQueueHandle device_1_cq_0_handle = GetCommandQueue(device_1, 0);
    CommandQueueHandle device_1_cq_1_handle = GetCommandQueue(device_1, 0);


    // Specify how the buffers are laid out inside local memory across both devices
    InterleavedBufferConfig buffer_config_device_0 = {
        .device = device_0,
        .size = dram_buffer_size_per_device,
        .page_size = dram_buffer_size_per_device,
        .buffer_type = tt_metal::BufferType::DRAM
    };

    InterleavedBufferConfig buffer_config_device_1 = {
        .device = device_1,
        .size = dram_buffer_size_per_device,
        .page_size = dram_buffer_size_per_device,
        .buffer_type = tt_metal::BufferType::DRAM
    };

    // Allocate Buffers used for IO in the workload
    //  ======== These Buffers live on Device 0 ========
    BufferHandle mul_src_0 = CreateBuffer(buffer_config_device_0);
    BufferHandle mul_src_1 = CreateBuffer(buffer_config_device_0);
    BufferHandle mul_dst = CreateBuffer(buffer_config_device_0);
    //  ======== These Buffers live on Device 1 ========
    BufferHandle add_src_0 = CreateBuffer(buffer_config_device_1);
    BufferHandle add_src_1 = CreateBuffer(buffer_config_device_1);
    BufferHandle add_dst = CreateBuffer(buffer_config_device_1);

    // Create Programs for both Devices. Each runs an independent binary operation (Mul on Device 0 and Add on Device 1)
    std::shared_ptr<Program> mul_program = create_binary_program(mul_src_0, mul_src_1, mul_dst, single_tile_size, num_tiles_per_device, BinaryOpType::MUL);
    std::shared_ptr<Program> add_program = create_binary_program(add_src_0, add_src_1, add_dst, single_tile_size, num_tiles_per_device, BInaryOpType::ADD);

    // Create randomized input data to drive the workload. Written to per-device DRAM buffers.
    std::vector<uint32_t> random_data_0 = create_random_vector_of_bfloat16(dram_buffer_size_per_device, 1, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> random_data_1 = create_random_vector_of_bfloat16(dram_buffer_size_per_device, 1, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> random_data_2 = create_random_vector_of_bfloat16(dram_buffer_size_per_device, 1, std::chrono::system_clock::now().time_since_epoch().count());

    // Setup output containers we will read data from the Devices into
    std::vector<uint32_t> mul_readback_data = {};
    std::vector<uint32_t> add_readback_data = {};

    // Data-Movement and Compute on Device 0. IO on CQ1, compute on CQ0. Use events to ensure ordering.
    std::shared_ptr<Event> device_0_write_event = std::make_shared<Event>();
    std::shared_ptr<Event> device_0_compute_event = std::make_shared<Event>();

    // Write inputs
    EnqueueWriteBuffer(device_0_cq_1_handle, mul_src_0, random_data_0);
    EnqueueWriteBuffer(device_0_cq_1_handle, mul_src_1, random_data_1);
    // Record that inputs were written
    EnqueueRecordEvent(device_0_cq_1_handle, device_0_write_event);
    // Wait until inputs were written
    EnqueueWaitForEvent(device_0_cq_0_handle, device_0_write_event);
    // Run compute
    EnqueueProgram(device_0_cq_0_handle, mul_program);
    // Record that compute was run and is completed
    EnqueueRecordEvent(device_0_cq_0_handle, device_0_compute_event);
    // Wait until compute has completed
    EnqueueWaitForEvent(device_0_cq_1_handle, device_0_compute_event);
    // Read outputs
    EnqueueReadBuffer(device_0_cq_1_handle, mul_dst, mul_readback_data);

    // Data-Movement and Compute on Device 1. IO and compute on CQ0. No need to use events to synchronize.
    // Write inputs
    EnqueueWriteBuffer(device_1_cq_0_handle, add_src_0, mul_readback_data);
    EnqueueWriteBuffer(device_1_cq_0_handle, add_src_1, random_data_2);
    // Run compute
    EnqueueMeshWorkload(device_1_cq_0_handle, add_program);
    // Read outputs
    EnqueueReadBuffer(device_1_cq_0_handle, add_dst, add_readback_data);

    CloseDevice(device_0);
    CloseDevice(device_1);
}
