// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "bfloat16.hpp"
#include "bfloat8.hpp"
#include "tilize_untilize.hpp"
#include "tt_backend_api_types.hpp"
#include "tt_metal/common/bfloat8.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

enum TEST_MODE {
    VERIFY,
    BENCHMARK,
};

std::vector<bfloat16> npu_unpack_bfp8_tile(
    const std::vector<float> &fp32_in_vec, bool unpack_to_bf16_in_reader_kernel, TEST_MODE mode) {
    Device *device = CreateDevice(0);
    std::vector<uint32_t> in_vec = pack_fp32_vec_as_bfp8_tiles(fp32_in_vec, true, false);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t bfp8_tile_size = 1088;
    constexpr uint32_t bf16_tile_size = 2 * 1024;

    std::shared_ptr<tt::tt_metal::Buffer> in_dram_buffer = CreateBuffer(
        {.device = device,
         .size = bfp8_tile_size,
         .page_size = bfp8_tile_size,
         .buffer_type = tt_metal::BufferType::DRAM});
    std::shared_ptr<tt::tt_metal::Buffer> out_dram_buffer = CreateBuffer(
        {.device = device,
         .size = bf16_tile_size,
         .page_size = bf16_tile_size,
         .buffer_type = tt_metal::BufferType::DRAM});

    auto in_dram_noc_coord = in_dram_buffer->noc_coordinates();
    auto out_dram_noc_coord = out_dram_buffer->noc_coordinates();
    uint32_t in_dram_noc_x = in_dram_noc_coord.x;
    uint32_t in_dram_noc_y = in_dram_noc_coord.y;
    uint32_t out_dram_noc_x = out_dram_noc_coord.x;
    uint32_t out_dram_noc_y = out_dram_noc_coord.y;

    constexpr uint32_t num_tiles = 1;
    constexpr uint32_t in0_cb_id = CB::c_in0;
    CircularBufferConfig in0_cb_config =
        CircularBufferConfig(num_tiles * bfp8_tile_size, {{in0_cb_id, tt::DataFormat::Bfp8_b}})
            .set_page_size(in0_cb_id, bfp8_tile_size);
    CBHandle cb_in = tt_metal::CreateCircularBuffer(program, core, in0_cb_config);

    constexpr uint32_t in1_cb_id = CB::c_in1;
    CircularBufferConfig in1_cb_config =
        CircularBufferConfig(num_tiles * bfp8_tile_size, {{in1_cb_id, tt::DataFormat::Bfp8_b}})
            .set_page_size(in1_cb_id, bfp8_tile_size);
    CBHandle cb_in1 = tt_metal::CreateCircularBuffer(program, core, in1_cb_config);

    constexpr uint32_t out_cb_id = CB::c_out0;
    CircularBufferConfig out_cb_config =
        CircularBufferConfig(num_tiles * bf16_tile_size, {{out_cb_id, tt::DataFormat::Float16_b}})
            .set_page_size(out_cb_id, bf16_tile_size);
    CBHandle cb_out = tt_metal::CreateCircularBuffer(program, core, out_cb_config);

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/unpack_bfp8_tile/kernels/dataflow/reader_unpack_bfp8_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/unpack_bfp8_tile/kernels/dataflow/writer_unpack_bfp8_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/unpack_bfp8_tile/kernels/compute/compute_unpack_bfp8_tile.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {},
        });

    EnqueueWriteBuffer(cq, in_dram_buffer, in_vec, false);

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {in_dram_buffer->address(), in_dram_noc_x, in_dram_noc_y, unpack_to_bf16_in_reader_kernel, mode});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {unpack_to_bf16_in_reader_kernel});
    SetRuntimeArgs(
        program, unary_writer_kernel_id, core, {out_dram_buffer->address(), out_dram_noc_x, out_dram_noc_y, mode});

    double total_time = 0;
    int host_loop_count = PROFILER_OP_SUPPORT_COUNT * kernel_profiler::PROFILER_L1_GUARANTEED_MARKER_COUNT;
    for (int i = 0; i < host_loop_count; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        EnqueueProgram(cq, program, true);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (i > 0) {  // Skip the first iteration for warm-up
            total_time += elapsed.count();
        }
    }
    double avg_time = total_time / (host_loop_count - 1);
    std::cout << "Average time: " << avg_time * 1000 << " milliseconds." << std::endl;

    Finish(cq);
    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, out_dram_buffer, result_vec, true);

    std::vector<bfloat16> bf16_vec = unpack_uint32_vec_into_bfloat16_vec(result_vec);

    tt_metal::detail::DumpDeviceProfileResults(device);
    CloseDevice(device);

    return bf16_vec;
}

std::vector<float> generate_random_float_vector(size_t size, float min_value, float max_value) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_value, max_value);
    for (auto &val : vec) val = dis(gen);
    return vec;
}

int main(int argc, char **argv) {
    bool unpack_to_bf16_in_reader_kernel = false;
    TEST_MODE mode = VERIFY;
    if (argc > 1) {
        unpack_to_bf16_in_reader_kernel = std::stoi(argv[1]);
    }
    if (argc > 2) {
        mode = static_cast<TEST_MODE>(std::stoi(argv[2]));
    }
    std::string mode_str = (mode == VERIFY) ? "VERIFY" : "BENCHMARK";
    std::string unpack_str = unpack_to_bf16_in_reader_kernel ? "true" : "false";
    std::cout << "Unpack to bf16 in reader kernel: " << unpack_str << std::endl;
    std::cout << "Mode: " << mode_str << std::endl;

    std::vector<float> fp32_in_vec = generate_random_float_vector(1024, 1, 2);
    std::vector<bfloat16> npu_bf16_vec = npu_unpack_bfp8_tile(fp32_in_vec, unpack_to_bf16_in_reader_kernel, mode);
    untilize(npu_bf16_vec, 32, 32);

    if (mode == VERIFY) {
        bool allclose = true;
        float rtol = 1e-01;  // relative tolerance
        float atol = 1e-03;  // absolute tolerance

        for (size_t i = 0; i < fp32_in_vec.size(); ++i) {
            auto cpu_fp32 = fp32_in_vec[i];
            auto npu_fp32 = npu_bf16_vec[i].to_float();
            if (std::abs(cpu_fp32 - npu_fp32) > (atol + rtol * std::abs(npu_fp32))) {
                std::cout << i << ": " << cpu_fp32 << " != " << npu_fp32 << std::endl;
                allclose = false;
            }
        }

        if (allclose) {
            std::cout << "CPU and NPU results are close enough." << std::endl;
        } else {
            std::cout << "CPU and NPU results differ." << std::endl;
        }
    }
}
