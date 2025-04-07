// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/allocator.hpp>

using namespace tt;

int main(int argc, char** argv) {
    bool pass = true;

    // We read back from the device, so require slow dispatch
    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);
        uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

        tt_metal::Program program = tt_metal::CreateProgram();
        CoreCoord core = {0, 0};

        tt_metal::KernelHandle kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/int_representation.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = {l1_unreserved_base}});

        for (unsigned ix = 0; ix != 8; ix++) {
            // Each test produces two values, which should be zero and
            // non-zero respectively.
            tt_metal::SetRuntimeArgs(program, kernel, core, {ix});

            tt_metal::detail::LaunchProgram(device, program);

            std::vector<uint32_t> results;
            tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, 2 * sizeof(int), results);
            bool ok = results[0] == 0 && results[1] != 0;
            log_info(LogVerif, "test {} results = {} & {}: {}", ix, results[0], results[1], ok ? "pass" : "fail");
            pass &= !ok;
        }

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
