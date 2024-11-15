// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <string>

#include "core_coord.hpp"
#include "logger.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/common/metal_soc_descriptor.h"
#include "tt_metal/impl/event/event.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/device/device.hpp"

constexpr uint32_t DEFAULT_ITERATIONS = 1000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 2;
constexpr uint32_t DEFAULT_PAGE_SIZE = 2048;
constexpr uint32_t DEFAULT_BATCH_SIZE_K = 512;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Test read/write bw and latency from host/dram/l1
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

uint32_t noc_idx = 0;
uint32_t proc_index = 0;
uint32_t read_barrier_period = 0;

uint32_t iterations_g = DEFAULT_ITERATIONS;
uint32_t warmup_iterations_g = DEFAULT_WARMUP_ITERATIONS;
CoreRange worker_g = {{0, 0}, {0, 0}};
CoreCoord src_worker_g = {0, 0};
CoreRange mcast_src_workers_g = {{0, 0}, {0, 0}};
uint32_t page_size_g;
uint32_t page_count_g;
uint32_t source_mem_g;
uint32_t dram_channel_g;
bool latency_g;
bool lazy_g;
bool time_just_finish_g;
bool read_one_packet_g;
bool page_size_as_runtime_arg_g; // useful particularly on GS multi-dram tests (multiply)
bool hammer_write_reg_g = false;
bool hammer_pcie_g = false;
bool hammer_pcie_type_g = false;
bool test_write = false;
bool linked = false;

void init(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up iterations before starting timer (default {}), ", DEFAULT_WARMUP_ITERATIONS);
        log_info(LogTest, "  -i: iterations (default {})", DEFAULT_ITERATIONS);
        log_info(LogTest, "  -bs: batch size in K of data to xfer in one iteration (default {}K)", DEFAULT_BATCH_SIZE_K);
        log_info(LogTest, "  -p: page size (default {})", DEFAULT_PAGE_SIZE);
        log_info(LogTest, "  -m: source mem, 0:PCIe, 1:DRAM, 2:L1, 3:ALL_DRAMs, 4:HOST_READ, 5:HOST_WRITE, 6:MULTICAST_WRITE (default 0:PCIe)");
        log_info(LogTest, "  -l: measure latency (default is bandwidth)");
        log_info(LogTest, "  -rx: X of core to issue read or write (default {})", 1);
        log_info(LogTest, "  -ry: Y of core to issue read or write (default {})", 0);
        log_info(LogTest, "  -sx: when reading from L1, X of core to read from. when issuing a multicast write, X of start core to write to. (default {})", 0);
        log_info(LogTest, "  -sy: when reading from L1, Y of core to read from. when issuing a multicast write, Y of start core to write to. (default {})", 0);
        log_info(LogTest, "  -tx: when issuing a multicast write, X of end core to write to (default {})", 0);
        log_info(LogTest, "  -ty: when issuing a multicast write, Y of end core to write to (default {})", 0);
        log_info(LogTest, "  -wr: issue unicast write instead of read (default false)");
        log_info(LogTest, "  -c: when reading from dram, DRAM channel (default 0)");
        log_info(LogTest, "  -f: time just the finish call (use w/ lazy mode) (default disabled)");
        log_info(LogTest, "  -o: use read_one_packet API.  restricts page size to 8K max (default {})", 0);
        log_info(LogTest, "  -z: enable dispatch lazy mode (default disabled)");
        log_info(LogTest, "-link: link mcast transactions");
        log_info(LogTest, " -hr: hammer write_reg while executing (for PCIe test)");
        log_info(LogTest, " -hp: hammer hugepage PCIe memory while executing (for PCIe test)");
        log_info(LogTest, " -hpt:hammer hugepage PCIe hammer type: 0:32bit writes 1:128bit non-temporal writes");
        log_info(LogTest, "  -psrta: pass page size as a runtime argument (default compile time define)");
        exit(0);
    }

    noc_idx = test_args::get_command_option_uint32(input_args, "-noc", 0);
    proc_index = test_args::get_command_option_uint32(input_args, "-proc", 0);
    read_barrier_period = test_args::get_command_option_uint32(input_args, "-rbp", 1);

    uint32_t core_x = test_args::get_command_option_uint32(input_args, "-rx", 1);
    uint32_t core_y = test_args::get_command_option_uint32(input_args, "-ry", 0);
    warmup_iterations_g = test_args::get_command_option_uint32(input_args, "-w", DEFAULT_WARMUP_ITERATIONS);
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    lazy_g = test_args::has_command_option(input_args, "-z");
    hammer_write_reg_g = test_args::has_command_option(input_args, "-hr");
    hammer_pcie_g = test_args::has_command_option(input_args, "-hp");
    hammer_pcie_type_g = test_args::get_command_option_uint32(input_args, "-hpt", 0);
    time_just_finish_g = test_args::has_command_option(input_args, "-f");
    source_mem_g = test_args::get_command_option_uint32(input_args, "-m", 0);
    uint32_t src_core_x = test_args::get_command_option_uint32(input_args, "-sx", 0);
    uint32_t src_core_y = test_args::get_command_option_uint32(input_args, "-sy", 0);
    uint32_t mcast_end_core_x = test_args::get_command_option_uint32(input_args, "-tx", 0);
    uint32_t mcast_end_core_y = test_args::get_command_option_uint32(input_args, "-ty", 0);
    dram_channel_g = test_args::get_command_option_uint32(input_args, "-c", 0);
    uint32_t size_bytes = test_args::get_command_option_uint32(input_args, "-bs", DEFAULT_BATCH_SIZE_K) * 1024;
    latency_g = test_args::has_command_option(input_args, "-l");
    page_size_g = test_args::get_command_option_uint32(input_args, "-p", DEFAULT_PAGE_SIZE);
    page_size_as_runtime_arg_g = test_args::has_command_option(input_args, "-psrta");
    read_one_packet_g = test_args::has_command_option(input_args, "-o");
    if (read_one_packet_g && page_size_g > 8192) {
        log_info(LogTest, "Page size must be <= 8K for read_one_packet\n");
        exit(-1);
    }
    page_count_g = size_bytes / page_size_g;

    test_write = test_args::has_command_option(input_args, "-wr");
    if (test_write && (source_mem_g != 2 && source_mem_g != 6)) {
        log_info(LogTest, "Writing only tested w/ L1 destination\n");
        exit(-1);
    }

    linked = test_args::has_command_option(input_args, "-link");

    worker_g = CoreRange({core_x, core_y}, {core_x, core_y});
    src_worker_g = {src_core_x, src_core_y};

    if (source_mem_g == 6)
    {
        if (mcast_end_core_x < src_core_x || mcast_end_core_y < src_core_y)
        {
            log_info(LogTest, "X of end core must be >= X of start core, Y of end core must be >= Y of start core");
            exit(-1);
        }

        mcast_src_workers_g = CoreRange({src_core_x, src_core_y}, {mcast_end_core_x, mcast_end_core_y});

        if (mcast_src_workers_g.intersects(worker_g)) {
            log_info(
                LogTest,
                "Multicast destination rectangle and core that issues the multicast cannot overlap - Multicast "
                "destination rectangle: {} Master core: {}", mcast_src_workers_g.str(), worker_g.start_coord.str());
            exit(-1);
        }
    }
}

#define CACHE_LINE_SIZE 64
void nt_memcpy(uint8_t *__restrict dst, const uint8_t * __restrict src, size_t n)
{
    size_t num_lines = n / CACHE_LINE_SIZE;

    size_t i;
    for (i = 0; i < num_lines; i++) {
        size_t j;
        for (j = 0; j < CACHE_LINE_SIZE / sizeof(__m128i); j++) {
            __m128i blk = _mm_loadu_si128((const __m128i *)src);
            /* non-temporal store */
            _mm_stream_si128((__m128i *)dst, blk);
            src += sizeof(__m128i);
            dst += sizeof(__m128i);
        }
        n -= CACHE_LINE_SIZE;
    }

    if (num_lines > 0)
        tt_driver_atomics::sfence();
}

int main(int argc, char **argv) {
    init(argc, argv);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        CommandQueue& cq = device->command_queue();

        tt_metal::Program program = tt_metal::CreateProgram();

        string src_mem;
        uint32_t noc_addr_x, noc_addr_y;
        uint64_t noc_mem_addr = 0;
        uint32_t dram_banked = 0;
        uint32_t issue_mcast = 0;
        uint32_t num_mcast_dests = mcast_src_workers_g.size();
        uint32_t mcast_noc_addr_end_x = 0;
        uint32_t mcast_noc_addr_end_y = 0;

        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
        void *host_pcie_base = (void*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
        uint64_t dev_pcie_base = tt::Cluster::instance().get_pcie_base_addr_from_device(device->id());
        uint64_t pcie_offset = 1024 * 1024 * 50;  // beyond where FD will write...maybe

        const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device->id());
        switch (source_mem_g) {
        case 0:
        default:
            {
                src_mem = "FROM_PCIE";
                vector<CoreCoord> pcie_cores = soc_d.get_pcie_cores();
                TT_ASSERT(pcie_cores.size() > 0);
                noc_addr_x = pcie_cores[0].x;
                noc_addr_y = pcie_cores[0].y;
                noc_mem_addr = dev_pcie_base + pcie_offset;
            }
            break;
        case 1:
            {
                src_mem = "FROM_DRAM";
                vector<CoreCoord> dram_cores = soc_d.get_dram_cores();
                TT_ASSERT(dram_cores.size() > dram_channel_g);
                noc_addr_x = dram_cores[dram_channel_g].x;
                noc_addr_y = dram_cores[dram_channel_g].y;
                log_info("{} dram channels total",dram_cores.size());
            }
            break;
        case 2:
            {
                src_mem = test_write ? "TO_L1" : "FROM_L1";
                CoreCoord w = device->physical_core_from_logical_core(src_worker_g, CoreType::WORKER);
                noc_addr_x = w.x;
                noc_addr_y = w.y;
            }
            break;
        case 3:
            {
                src_mem = "FROM_ALL_DRAMS";
                dram_banked = 1;
                noc_addr_x = -1; // unused
                noc_addr_y = -1; // unused
                noc_mem_addr = 0;
            }
            break;
        case 4:
            {
                src_mem = "FROM_L1_TO_HOST";
                log_info(LogTest, "Host bw test overriding page_count to 1");
                CoreCoord w = device->physical_core_from_logical_core(src_worker_g, CoreType::WORKER);
                page_count_g = 1;
                noc_addr_x = w.x;
                noc_addr_y = w.y;
            }
            break;
        case 5:
            {
                src_mem = "FROM_HOST_TO_L1";
                log_info(LogTest, "Host bw test overriding page_count to 1");
                CoreCoord w = device->physical_core_from_logical_core(src_worker_g, CoreType::WORKER);
                page_count_g = 1;
                noc_addr_x = w.x;
                noc_addr_y = w.y;
            }
            break;
        case 6:
            {
                src_mem = "FROM_L1_TO_MCAST";
                issue_mcast = 1;
                CoreCoord start = device->physical_core_from_logical_core(mcast_src_workers_g.start_coord, CoreType::WORKER);
                CoreCoord end = device->physical_core_from_logical_core(mcast_src_workers_g.end_coord, CoreType::WORKER);
                noc_addr_x = start.x;
                noc_addr_y = start.y;
                mcast_noc_addr_end_x = end.x;
                mcast_noc_addr_end_y = end.y;
                test_write = true;
            }
            break;
        }

        std::map<string, string> defines = {
            {"ITERATIONS", std::to_string(iterations_g)},
            {"PAGE_COUNT", std::to_string(page_count_g)},
            {"LATENCY", std::to_string(latency_g)},
            {"NOC_ADDR_X", std::to_string(noc_addr_x)},
            {"NOC_ADDR_Y", std::to_string(noc_addr_y)},
            {"NOC_MEM_ADDR", std::to_string(noc_mem_addr)},
            {"READ_ONE_PACKET", std::to_string(read_one_packet_g)},
            {"DRAM_BANKED", std::to_string(dram_banked)},
            {"ISSUE_MCAST", std::to_string(issue_mcast)},
            {"WRITE", std::to_string(test_write)},
            {"LINKED", std::to_string(linked)},
            {"NUM_MCAST_DESTS", std::to_string(num_mcast_dests)},
            {"MCAST_NOC_END_ADDR_X", std::to_string(mcast_noc_addr_end_x)},
            {"MCAST_NOC_END_ADDR_Y", std::to_string(mcast_noc_addr_end_y)},
            {"READ_BARRIER_PERIOD", std::to_string(read_barrier_period)}
        };
        //std::cout << "\n\n DEFINES:\n";
        //for (auto [k,v] : defines){
        //    std::cout << k << " : " << v << "\n";
        //}
        if (!page_size_as_runtime_arg_g) {
            defines.insert(std::pair<string, string>("PAGE_SIZE", std::to_string(page_size_g)));
        }

        tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(page_size_g * page_count_g, {{0, tt::DataFormat::Float32}})
            .set_page_size(0, page_size_g);
        auto cb = tt_metal::CreateCircularBuffer(program, worker_g, cb_config);

        tt_metal::NOC noc_type = (noc_idx == 1) ? tt_metal::NOC::NOC_1 : tt_metal::NOC::NOC_0;
        tt_metal::DataMovementProcessor proc_type = 
            (proc_index == 1) ? 
            tt_metal::DataMovementProcessor::RISCV_1 : 
            tt_metal::DataMovementProcessor::RISCV_0;

        auto dm0 = tt_metal::CreateKernel(
                                          program,
                                          "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/bw_and_latency.cpp",
                                          worker_g,
                                          tt_metal::DataMovementConfig{
                                            .processor = proc_type, 
                                            .noc = noc_type, 
                                            .defines = defines});
        if (page_size_as_runtime_arg_g) {
            tt_metal::SetRuntimeArgs(program, dm0, worker_g.start_coord, {page_size_g});
        }

        std::shared_ptr<Event> sync_event = std::make_shared<Event>();

        vector<uint32_t>blank(page_size_g / sizeof(uint32_t));
        std::chrono::duration<double> elapsed_seconds;
        if (source_mem_g < 4 || source_mem_g == 6) {
            // Cache stuff
            for (int i = 0; i < warmup_iterations_g; i++) {
                EnqueueProgram(cq, program, false);
            }
            Finish(cq);

            auto start = std::chrono::system_clock::now();
            EnqueueProgram(cq, program, false);
            Finish(cq);
            auto end = std::chrono::system_clock::now();

            elapsed_seconds = (end-start);
        }

        auto latency = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_seconds).count() / (page_count_g * iterations_g);

        CoreCoord dst_coord = device->physical_core_from_logical_core(*(worker_g.begin()), CoreType::WORKER);
        //log_info("{:5s} {:6s} {:6s} {:2s} {:2s} {:8s} \n", "NOC", "RISC", "MODE", "SX", "SY", "LATENCY");
        float ns_etime = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_seconds).count();
        float bytes_per_ns = (page_count_g * page_size_g * iterations_g) / ns_etime;
        float cycles_per_iter = ns_etime / iterations_g;
        float read_barriers_per_iter = float(page_count_g) / read_barrier_period;
        float cycles_per_barrier = float(cycles_per_iter) / read_barriers_per_iter;

        log_info("METRIC: {:5s} {:6s} {:5s}  {},{}  {},{}  {:7.1f} cycles  {:5.1f} GB/s RBPP {:.2f}\n", 
                (noc_idx == 1) ? "NOC_1" : "NOC_0",
                (proc_index == 1) ? "BRISC" : "NCRISC",
                test_write ? "WRITE" : "READ", 
                dst_coord.x, dst_coord.y, 
                noc_addr_x, noc_addr_y, 
                cycles_per_barrier,
                bytes_per_ns, 
                read_barriers_per_iter);

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }
}
