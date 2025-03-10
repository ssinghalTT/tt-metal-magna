// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bos_utils.hpp>

#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;

/*
 * 1. Host creates one vector of data.
 * 2. Device eltwise performs a unary SFPU operation on the data.
 * 3. Read result back and compare to golden.
 * */

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        /*
         * Silicon accelerator setup
         */
        constexpr int device_id = 1;
        IDevice* device = CreateDevice(device_id);

        /*
         * Setup program to execute along with its buffers and kernels to use
         */
        CommandQueue& cq = device->command_queue();

        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t single_tile_size_src = 2 * 1024;
        constexpr uint32_t single_tile_size_dst = 1024;
        constexpr uint32_t num_tiles = 64;
        constexpr uint32_t dram_buffer_size_src = single_tile_size_src * num_tiles;
        constexpr uint32_t dram_buffer_size_dst = single_tile_size_dst * num_tiles;

        tt_metal::InterleavedBufferConfig dram_config_src{
            .device = device,
            .size = dram_buffer_size_src,
            .page_size = dram_buffer_size_src,
            .buffer_type = tt_metal::BufferType::DRAM};
        
        tt_metal::InterleavedBufferConfig dram_config_dst{
            .device = device,
            .size = dram_buffer_size_dst,
            .page_size = dram_buffer_size_dst,
            .buffer_type = tt_metal::BufferType::DRAM};

        std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config_src);
        const uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

        std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config_dst);
        const uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();
        // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank
        uint32_t src0_bank_id = 0;
        uint32_t dst_bank_id = 0;

        /*
         * Use circular buffers to set input and output buffers that the
         * compute engine will use.
         */
        constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
        constexpr uint32_t num_input_tiles = 2;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size_src, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size_src);
        CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
        constexpr uint32_t num_output_tiles = 2;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * single_tile_size_dst, {{output_cb_index, tt::DataFormat::UInt8}})
                .set_page_size(output_cb_index, single_tile_size_dst);
        CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        /*
         * Specify data movement kernels for reading/writing data to/from
         * DRAM.
         */
        KernelHandle unary_reader_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        KernelHandle unary_writer_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        /*
         * Set the parameters that the compute kernel will use.
         */
        std::vector<uint32_t> compute_kernel_args = {num_tiles, 1};

        constexpr bool math_approx_mode = false;

        /*
         * Use defines to control the operations to execute in the eltwise_sfpu
         * compute kernel.
         */
        const std::map<std::string, std::string> sfpu_defines = {{"SFPU_OP_TYPECAST_INCLUDE", "1"}};

        KernelHandle eltwise_sfpu_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/bos/kernels/kernel_sfpu_typecasting_bfloat16_uint8.cpp",
            core,
            ComputeConfig{
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args,
                .defines = sfpu_defines,
            });

        /*
         * Create source data and write to DRAM.
         */
        // Create a linear vector, where src[0] = 1.1f, src[i+1] = src[i] + 1.1f
        std::vector<uint32_t> src0_vec = bos_create_linear_vector_of_bfloat16(dram_buffer_size_src, 1.11f, 1.11f);

        EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);

        /*
         * Configure program and runtime kernel arguments, then execute.
         */
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src0_dram_buffer->address(),
                src0_bank_id,
                num_tiles,
            });

        SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_bank_id, num_tiles});

        EnqueueProgram(cq, program, false);
        Finish(cq);

        /*
         * Read the result and compare to a golden result. Record pass/fail
         * and teardown.
         */
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

        auto transform_to_golden = [](const bfloat16& a) { return bfloat16(std::round(a.to_float())); };
        std::vector<uint32_t> golden_vec = bos_casting_bfloat16_vec_to_uint8_vec(src0_vec, transform_to_golden);

        constexpr float abs_tolerance = 10.02f;
        constexpr float rel_tolerance = 10.02f;
        auto comparison_function = [](const float a, const float b) {
            return bos_is_close(a, b, rel_tolerance, abs_tolerance);
        };

        pass &= bos_uint8_vector_comparison(golden_vec, result_vec, comparison_function);

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
