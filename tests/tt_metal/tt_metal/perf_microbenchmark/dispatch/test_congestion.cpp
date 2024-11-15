// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "core_coord.hpp"
#include "logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

using std::vector;
using namespace tt;

int main(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    auto concurrent_readers = test_args::get_command_option_uint32(input_args, "-n", 3);
    auto packet_size = test_args::get_command_option_uint32(input_args, "-p", 8192);
    log_info("concurrent readers: {}", concurrent_readers);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        CommandQueue &cq = device->command_queue();
        tt_metal::Program program = tt_metal::CreateProgram();

        //-- setup worker coordinates -----------------------------------------//
        CoreRange cores = CoreRange({0, 0}, {0, (concurrent_readers-1)});

        //-- setup defines ---------------------------------------------------//
        constexpr uint32_t ITERATIONS = 1'000;
        constexpr uint32_t PAGE_COUNT = 1'000;
        const uint32_t PAGE_SIZE = packet_size;

        const std::map<string, string> common_defines = {
            {"ITERATIONS", std::to_string(ITERATIONS)},
            {"PAGE_COUNT", std::to_string(PAGE_COUNT)},
            {"PAGE_SIZE", std::to_string(PAGE_SIZE)}};

        //-- program worker cores --------------------------------------------//
        tt_metal::NOC noc_type = tt_metal::NOC::NOC_1;
        tt_metal::DataMovementProcessor proc_type = tt_metal::DataMovementProcessor::RISCV_0;

        auto kernel_cpp_file = "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/congested_read.cpp";

        for (auto logical_loc : cores) {
            auto phys_loc = device->physical_core_from_logical_core(logical_loc, CoreType::WORKER);

            auto src_loc = device->physical_core_from_logical_core(
                {logical_loc.x, concurrent_readers-1}, CoreType::WORKER);
            std::vector<uint32_t> runtime_args = {src_loc.x, src_loc.y};

            log_info(
                "Setting up reader at {},{} reading from {},{} ... ", phys_loc.x, phys_loc.y, src_loc.x, src_loc.y);
            auto kernel_handle = tt_metal::CreateKernel(
                program,
                kernel_cpp_file,
                logical_loc,
                tt_metal::DataMovementConfig{
                    .processor = proc_type, .noc = tt_metal::NOC::NOC_1, .defines = common_defines});
            tt_metal::SetRuntimeArgs(program, kernel_handle, logical_loc, runtime_args);
        }

        //-- warmup ----------------------------------------------------------//
        EnqueueProgram(cq, program, false);
        Finish(cq);

        //-- run -------------------------------------------------------------//
        auto start = std::chrono::system_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto end = std::chrono::system_clock::now();
        float etime_ns = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        log_info("            elapsed time: {:.1f} ns", etime_ns);
        log_info(" elapsed time per packet: {:.1f} ns", etime_ns / (ITERATIONS*PAGE_COUNT));
        //log_info("   oneway latency: {:.1f} ns", etime_ns / (2 * ITERATIONS));
        //// measured by profiling loop with local semaphore increment
        //constexpr float loop_overhead_ns = 13.5;
        //log_info("--------------------------------------------", etime_ns / (2 * ITERATIONS));
        //log_info(
        //    "   oneway latency, _minus loop overhead_: {:.1f} ns", (etime_ns / (2 * ITERATIONS)) - loop_overhead_ns);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        log_fatal(e.what());
    }
}
