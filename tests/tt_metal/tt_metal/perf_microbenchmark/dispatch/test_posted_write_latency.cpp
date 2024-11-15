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
    auto src_x = test_args::get_command_option_uint32(input_args, "-sx", 0);
    auto src_y = test_args::get_command_option_uint32(input_args, "-sy", 0);
    auto dst_x = test_args::get_command_option_uint32(input_args, "-dx", 1);
    auto dst_y = test_args::get_command_option_uint32(input_args, "-dy", 0);
    log_info("src_xy:{},{}  dst_xy:{},{}", src_x, src_y, dst_x, dst_y);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        CommandQueue &cq = device->command_queue();
        tt_metal::Program program = tt_metal::CreateProgram();

        //-- setup worker coordinates -----------------------------------------//
        CoreCoord src_worker_g = {src_x, src_y};
        CoreCoord dst_worker_g = {dst_x, dst_y};
        CoreRangeSet cores = std::set<CoreRange>({CoreRange(src_worker_g), CoreRange(dst_worker_g)});

        auto [src_phys_addr_x, src_phys_addr_y] =
            device->physical_core_from_logical_core(src_worker_g, CoreType::WORKER);
        auto [dst_phys_addr_x, dst_phys_addr_y] =
            device->physical_core_from_logical_core(dst_worker_g, CoreType::WORKER);

        //-- setup defines ---------------------------------------------------//
        constexpr uint32_t ITERATIONS = 1'000'000;
        const std::map<string, string> common_defines = {{"ITERATIONS", std::to_string(ITERATIONS)}};
        std::map<string, string> src_defines = {
            {"OTHER_ADDR_X", std::to_string(dst_phys_addr_x)},
            {"OTHER_ADDR_Y", std::to_string(dst_phys_addr_y)},
            {"START", std::to_string(1)}};
        std::map<string, string> dst_defines = {
            {"OTHER_ADDR_X", std::to_string(src_phys_addr_x)},
            {"OTHER_ADDR_Y", std::to_string(src_phys_addr_y)},
            {"START", std::to_string(0)}};
        src_defines.insert(common_defines.begin(), common_defines.end());
        dst_defines.insert(common_defines.begin(), common_defines.end());

        //-- setup semaphore -------------------------------------------------//
        std::vector<uint32_t> runtime_args(1);
        runtime_args.push_back(tt_metal::CreateSemaphore(program, cores, INVALID));
        log_info("semaphore id(s) = {}\n", runtime_args[0]);

        //-- program worker cores --------------------------------------------//
        tt_metal::NOC noc_type = tt_metal::NOC::NOC_1;
        tt_metal::DataMovementProcessor proc_type = tt_metal::DataMovementProcessor::RISCV_0;

        auto kernel_cpp_file = "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/sema_inc.cpp";

        auto dm0 = tt_metal::CreateKernel(
            program,
            kernel_cpp_file,
            dst_worker_g,
            tt_metal::DataMovementConfig{.processor = proc_type, .noc = tt_metal::NOC::NOC_1, .defines = dst_defines});
        tt_metal::SetRuntimeArgs(program, dm0, dst_worker_g, runtime_args);

        auto dm1 = tt_metal::CreateKernel(
            program,
            kernel_cpp_file,
            src_worker_g,
            tt_metal::DataMovementConfig{.processor = proc_type, .noc = tt_metal::NOC::NOC_0, .defines = src_defines});
        tt_metal::SetRuntimeArgs(program, dm1, src_worker_g, runtime_args);

        //-- warmup ----------------------------------------------------------//
        EnqueueProgram(cq, program, false);
        Finish(cq);

        //-- run -------------------------------------------------------------//
        auto start = std::chrono::system_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto end = std::chrono::system_clock::now();
        float etime_ns = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        log_info("     elapsed time: {:.1f} ns", etime_ns);
        log_info("roundtrip latency: {:.1f} ns", etime_ns / ITERATIONS);
        log_info("   oneway latency: {:.1f} ns", etime_ns / (2 * ITERATIONS));
        // measured by profiling loop with local semaphore increment
        constexpr float loop_overhead_ns = 13.5;
        log_info("--------------------------------------------", etime_ns / (2 * ITERATIONS));
        log_info(
            "   oneway latency, _minus loop overhead_: {:.1f} ns", (etime_ns / (2 * ITERATIONS)) - loop_overhead_ns);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        log_fatal(e.what());
    }
}
