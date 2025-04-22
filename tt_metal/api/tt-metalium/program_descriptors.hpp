// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/trivial_static_vector.hpp>

#include <umd/device/tt_core_coordinates.h>

#include <boost/container/small_vector.hpp>

#include <optional>

namespace tt::tt_metal {

struct Tile;
class Buffer;
namespace experimental {
class GlobalCircularBuffer;
}  // namespace experimental

struct TileDescriptor {
    TileDescriptor() = default;
    TileDescriptor(const Tile& tile);
    TileDescriptor(uint32_t height, uint32_t width, bool transpose) :
        height(height), width(width), transpose(transpose) {}

    uint32_t height = constants::TILE_HEIGHT;
    uint32_t width = constants::TILE_WIDTH;
    bool transpose = false;
};

struct CBFormatDescriptor {
    uint8_t buffer_index = 0;
    tt::DataFormat data_format = tt::DataFormat::Float32;
    uint32_t page_size = 0;
    std::optional<TileDescriptor> tile;
};

struct CBDescriptor {
    using FormatDescriptors = boost::container::small_vector<CBFormatDescriptor, 1>;

    uint32_t total_size = 0;
    CoreRangeVector core_ranges;
    FormatDescriptors format_descriptors;
    FormatDescriptors remote_format_descriptors;

    Buffer* buffer = nullptr;
    const experimental::GlobalCircularBuffer* global_circular_buffer = nullptr;
};

struct SemaphoreDescriptor {
    CoreType core_type = CoreType::WORKER;
    CoreRangeVector core_ranges;
    uint32_t initial_value = 0;
};

struct ReaderConfigDescriptor {};
struct WriterConfigDescriptor {};
struct DataMovementConfigDescriptor {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
    NOC noc = NOC::RISCV_0_default;
    NOC_MODE noc_mode = NOC_MODE::DM_DEDICATED_NOC;
};
struct ComputeConfigDescriptor {
    using UnpackToDestModes = std::vector<UnpackToDestMode>;

    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    UnpackToDestModes unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
};
struct EthernetConfigDescriptor {
    Eth eth_mode = Eth::SENDER;
    NOC noc = NOC::NOC_0;
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
};

struct KernelDescriptor {
    using CompileTimeArgs = tt::stl::trivial_static_vector<uint32_t, 32>;
    using Defines = boost::container::small_vector<std::pair<std::string, std::string>, 16>;
    using CoreRuntimeArgs = tt::stl::trivial_static_vector<uint32_t, 32>;
    using RuntimeArgs = boost::container::small_vector<boost::container::small_vector<CoreRuntimeArgs, 8>, 8>;
    using CommonRuntimeArgs = tt::stl::trivial_static_vector<uint32_t, 32>;
    using ConfigDescriptor = std::variant<
        ReaderConfigDescriptor,
        WriterConfigDescriptor,
        DataMovementConfigDescriptor,
        ComputeConfigDescriptor,
        EthernetConfigDescriptor>;
    enum class SourceType { FILE_PATH, SOURCE_CODE };

    std::string kernel_source;
    SourceType source_type = SourceType::FILE_PATH;

    CoreRangeVector core_ranges;
    CompileTimeArgs compile_time_args;
    Defines defines;

    RuntimeArgs runtime_args;
    CommonRuntimeArgs common_runtime_args;

    std::optional<KernelBuildOptLevel> opt_level = std::nullopt;

    ConfigDescriptor config;

    void reserve_runtime_args();
};

struct ProgramDescriptor {
    using KernelDescriptors = boost::container::small_vector<KernelDescriptor, 3>;
    using SemaphoreDescriptors = boost::container::small_vector<SemaphoreDescriptor, 3>;
    using CBDescriptors = boost::container::small_vector<CBDescriptor, 5>;

    KernelDescriptors kernels;
    SemaphoreDescriptors semaphores;
    CBDescriptors cbs;

    uint32_t add_semaphore(CoreRangeVector core_ranges, uint32_t initial_value, CoreType core_type = CoreType::WORKER);
    size_t calculate_program_hash() const;
};

struct MeshWorkloadDescriptor {
    using ProgramDescriptors =
        boost::container::small_vector<std::pair<distributed::MeshCoordinateRange, ProgramDescriptor>, 1>;

    ProgramDescriptors programs;

    size_t calculate_program_hash() const;
};

}  // namespace tt::tt_metal
