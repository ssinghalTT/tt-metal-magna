// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Utility functions and data structures related to tt-metal kernel profiler's noc tracing feature

#include <tuple>
#include <optional>
#include <map>
#include <vector>
#include <utility>
#include <nlohmann/json.hpp>

#include "tt_cluster.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "tt_metal.hpp"

namespace tt {

namespace tt_metal {

// Pre-computes and stores routing information for 1D Fabric topologies
// (Ring/Linear).  This class builds a lookup table during initialization based
// on the cluster's Fabric configuration *if it detects a 1D Ring or Linear
// topology*.
//
// If the Fabric topology is *NOT* 1D, the lookup object remains empty
// (default-constructed).
//
// The lookup is safe to use if default constructed or unpopulated; it simply
// returns null optionals for all lookups.
//
// The main problem this solves is precomputing the relationship between ERISC
// cores and fabric channels, and the on-chip forwarding paths between different
// channels. The source-of-truth is (mostly) the control_plane.hpp and
// tt-cluster.hpp APIs.  The stored information is mainly intended to be used
// during dumpResults() to interpret and elaborate the relatively limited
// information captured for each fabric event (usually just destination coord
// and hop count) into a complete set of piecewise routes from source of fabric
// packet --> end destination.

class FabricRoutingLookup {
public:
    // both of these are keyed by physical chip id!
    using EthCoreToChannelMap = std::map<std::tuple<chip_id_t, CoreCoord>, tt::tt_fabric::chan_id_t>;

    // Default constructor for cases where lookup is not built (e.g., non-1D fabric)
    FabricRoutingLookup() = default;

    FabricRoutingLookup(const IDevice* device) {
        using namespace tt::tt_fabric;

        Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

        // get sorted list of all physical chip ids
        auto physical_chip_id_set = cluster.user_exposed_chip_ids();
        std::vector<chip_id_t> physical_chip_ids(physical_chip_id_set.begin(), physical_chip_id_set.end());
        std::sort(physical_chip_ids.begin(), physical_chip_ids.end());

        for (chip_id_t chip_id_src : physical_chip_ids) {
            if (device->is_mmio_capable() && (cluster.get_cluster_type() == tt::ClusterType::TG)) {
                // skip lauching on gateways for TG
                continue;
            }

            // NOTE: soc desc is for chip_id_src, not device->id()
            const auto& soc_desc = cluster.get_soc_desc(chip_id_src);
            // Build a mapping of (eth_core --> eth_chan)
            for (chan_id_t eth_chan = 0; eth_chan < soc_desc.get_num_eth_channels(); eth_chan++) {
                auto eth_physical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::PHYSICAL);
                eth_core_to_channel_lookup_.emplace(std::make_tuple(chip_id_src, eth_physical_core), eth_chan);
            }
        }
    }

    // lookup APIs
    std::optional<tt::tt_fabric::chan_id_t> getRouterEthCoreToChannelLookup(
        chip_id_t chip_id, CoreCoord eth_router_core_coord) const {
        auto it = eth_core_to_channel_lookup_.find(std::make_tuple(chip_id, eth_router_core_coord));
        if (it != eth_core_to_channel_lookup_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    EthCoreToChannelMap eth_core_to_channel_lookup_;
};

inline void dumpClusterCoordinatesAsJson(const std::filesystem::path& filepath) {
    Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    nlohmann::ordered_json cluster_json;
    cluster_json["physical_chip_to_eth_coord"] = nlohmann::ordered_json();
    for (auto& [chip_id, eth_core] : cluster.get_user_chip_ethernet_coordinates()) {
        eth_coord_t eth_coord = eth_core;
        auto& entry = cluster_json["physical_chip_to_eth_coord"][std::to_string(chip_id)];
        entry["rack"] = eth_coord.rack;
        entry["shelf"] = eth_coord.shelf;
        entry["x"] = eth_coord.x;
        entry["y"] = eth_coord.y;
    }

    std::ofstream cluster_json_ofs(filepath);
    if (cluster_json_ofs.is_open()) {
        cluster_json_ofs << cluster_json.dump(2);
    } else {
        log_error("Failed to open file '{}' for dumping cluster coordinate map", filepath.string());
    }
}

}  // namespace tt_metal
}  // namespace tt
