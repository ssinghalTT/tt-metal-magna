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
    using ForwardingRoute = std::pair<CoreCoord, CoreCoord>;
    using FullFabricRoute = std::vector<std::pair<chip_id_t, ForwardingRoute>>;
    using RoutingPathMap = std::map<std::tuple<chip_id_t, CoreCoord, int>, FullFabricRoute>;

    // Default constructor for cases where lookup is not built (e.g., non-1D fabric)
    FabricRoutingLookup() = default;

    FabricRoutingLookup(const IDevice* device) {
        using namespace tt::tt_fabric;

        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

        // establish that we have a 1D fabric, otherwise bail
        tt::tt_metal::FabricConfig fabric_config = cluster.get_fabric_config();
        if (!tt::tt_fabric::is_1d_fabric_config(fabric_config)) {
            log_info("Skipping fabric routing lookup build; topology is not 1D Ring or Linear");
            // Return default-constructed (empty) lookup object
            return;
        }

        const tt::tt_fabric::ControlPlane* control_plane = cluster.get_control_plane();
        TT_ASSERT(control_plane != nullptr);

        // get sorted list of all physical chip ids
        auto physical_chip_id_set = cluster.user_exposed_chip_ids();
        std::vector<chip_id_t> physical_chip_ids(physical_chip_id_set.begin(), physical_chip_id_set.end());
        std::sort(physical_chip_ids.begin(), physical_chip_ids.end());

        // PHASE I - Build various lookups for chip,direction --> neighbour and eth_core --> forwarding route
        // based on create_and_compile_1d_fabric_program() from topology.cpp
        std::map<std::pair<chip_id_t, tt_fabric::RoutingDirection>, chip_id_t>
            chip_neighbors;  // indexed by physical chip id
        const std::vector<tt_fabric::RoutingDirection> routing_directions = {
            tt_fabric::RoutingDirection::N,
            tt_fabric::RoutingDirection::S,
            tt_fabric::RoutingDirection::E,
            tt_fabric::RoutingDirection::W};
        std::map<std::pair<chip_id_t, CoreCoord>, CoreCoord> forwarding_pairs;
        std::map<std::pair<chip_id_t, tt_fabric::RoutingDirection>, std::set<tt_fabric::chan_id_t>>
            active_fabric_eth_channels;

        for (chip_id_t chip_id_src : physical_chip_ids) {
            if (device->is_mmio_capable() &&
                (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG)) {
                // skip lauching on gateways for TG
                continue;
            }

            // build mapping of routing_dir -> [eth_channel]
            // build mapping of routing_dir -> neighbor_chip_id
            std::pair<tt_fabric::mesh_id_t, chip_id_t> mesh_chip_id =
                control_plane->get_mesh_chip_id_from_physical_chip_id(chip_id_src);
            for (const auto& direction : routing_directions) {
                auto active_eth_chans = control_plane->get_active_fabric_eth_channels_in_direction(
                    mesh_chip_id.first, mesh_chip_id.second, direction);
                if (active_eth_chans.empty()) {
                    continue;
                }

                // this returns a list of *mesh chip ids*, not physical!
                auto neighbors =
                    control_plane->get_intra_chip_neighbors(mesh_chip_id.first, mesh_chip_id.second, direction);
                if (neighbors.empty()) {
                    continue;
                }

                // assume same neighbor per direction
                std::pair<tt_fabric::mesh_id_t, chip_id_t> neighbor_mesh_chip_id = {mesh_chip_id.first, neighbors[0]};
                chip_neighbors[{chip_id_src, direction}] =
                    control_plane->get_physical_chip_id_from_mesh_chip_id(neighbor_mesh_chip_id);
                active_fabric_eth_channels[{chip_id_src, direction}] = active_eth_chans;
            }

            // NOTE: soc desc is for chip_id_src, not device->id()
            const auto& soc_desc = cluster.get_soc_desc(chip_id_src);
            // Build a mapping of (chip_id, eth_core) -> eth_chan
            for (const auto& [_, eth_channels] : active_fabric_eth_channels) {
                for (const auto& eth_chan : eth_channels) {
                    auto eth_physical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::PHYSICAL);
                    eth_core_to_channel_lookup_.emplace(std::make_tuple(chip_id_src, eth_physical_core), eth_chan);
                }
            }

            // infer forwarding paths if neighbors exist in both directions
            for (const auto& direction_pair : std::vector<std::pair<RoutingDirection, RoutingDirection>>{
                     {RoutingDirection::E, RoutingDirection::W}, {RoutingDirection::N, RoutingDirection::S}}) {
                auto [dir1, dir2] = direction_pair;
                if (chip_neighbors.contains({chip_id_src, dir1}) && chip_neighbors.contains({chip_id_src, dir2})) {
                    auto& eth_chans_dir1 = active_fabric_eth_channels.at({chip_id_src, dir1});
                    auto& eth_chans_dir2 = active_fabric_eth_channels.at({chip_id_src, dir2});

                    auto eth_chans_dir1_it = eth_chans_dir1.begin();
                    auto eth_chans_dir2_it = eth_chans_dir2.begin();

                    // since tunneling cores are not guaraneteed to be reserved on the same routing plane, iterate
                    // through the sorted eth channels in both directions
                    while (eth_chans_dir1_it != eth_chans_dir1.end() && eth_chans_dir2_it != eth_chans_dir2.end()) {
                        auto eth_chan_dir1 = *eth_chans_dir1_it;
                        auto eth_chan_dir2 = *eth_chans_dir2_it;

                        auto eth_physical_core_dir1 =
                            soc_desc.get_eth_core_for_channel(eth_chan_dir1, CoordSystem::PHYSICAL);
                        auto eth_physical_core_dir2 =
                            soc_desc.get_eth_core_for_channel(eth_chan_dir2, CoordSystem::PHYSICAL);

                        forwarding_pairs[{chip_id_src, eth_physical_core_dir1}] = eth_physical_core_dir2;
                        forwarding_pairs[{chip_id_src, eth_physical_core_dir2}] = eth_physical_core_dir1;

                        log_info(
                            "Connecting Chan {} @ {},{} to Chan {} @ {},{} (BIDIRECTIONALLY)",
                            eth_chan_dir1,
                            eth_physical_core_dir1.x,
                            eth_physical_core_dir1.y,
                            eth_chan_dir2,
                            eth_physical_core_dir2.x,
                            eth_physical_core_dir2.y);

                        eth_chans_dir1_it++;
                        eth_chans_dir2_it++;
                    }
                }
            }
        }

        // PHASE II - Build routing path map (precomputed route set including all forwarding for any chip to chip
        // transfer)
        for (chip_id_t chip_id_src : physical_chip_ids) {
            if (device->is_mmio_capable() &&
                (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG)) {
                // skip lauching on gateways for TG
                continue;
            }
            std::pair<tt_fabric::mesh_id_t, chip_id_t> mesh_chip_id_src =
                control_plane->get_mesh_chip_id_from_physical_chip_id(chip_id_src);

            for (chip_id_t chip_id_dst : physical_chip_ids) {
                if (chip_id_src == chip_id_dst) {
                    continue;
                }
                std::pair<tt_fabric::mesh_id_t, chip_id_t> mesh_chip_id_dst =
                    control_plane->get_mesh_chip_id_from_physical_chip_id(chip_id_dst);

                for (auto direction : routing_directions) {
                    for (auto eth_chan : active_fabric_eth_channels[{chip_id_src, direction}]) {
                        const metal_SocDescriptor& src_soc_desc = cluster.get_soc_desc(chip_id_src);
                        auto src_router_eth_core =
                            src_soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::PHYSICAL);
                        std::vector<std::pair<chip_id_t, tt_fabric::chan_id_t>> route = control_plane->get_fabric_route(
                            mesh_chip_id_src.first,
                            mesh_chip_id_src.second,
                            mesh_chip_id_dst.first,
                            mesh_chip_id_dst.second,
                            eth_chan);

                        size_t hop_count = route.size();

                        // get all intermediate forwarding routes, iterate all but the last element in route
                        FabricRoutingLookup::FullFabricRoute all_routes;
                        all_routes.push_back({chip_id_src, {{0, 0}, src_router_eth_core}});
                        for (size_t i = 0; i < route.size(); ++i) {
                            auto& [route_chip_id, route_eth_chan] = route[i];
                            const metal_SocDescriptor& route_soc_desc = cluster.get_soc_desc(route_chip_id);
                            auto route_eth_physical_core =
                                route_soc_desc.get_eth_core_for_channel(route_eth_chan, CoordSystem::PHYSICAL);
                            if (i == route.size() - 1) {
                                // last leg of route, last coordinate is placeholder set to {0,0}
                                FabricRoutingLookup::ForwardingRoute forwarding_route = {
                                    route_eth_physical_core, {0, 0}};
                                all_routes.push_back({route_chip_id, forwarding_route});
                            } else if (forwarding_pairs.contains({route_chip_id, route_eth_physical_core})) {
                                // intermediate step, add forwarding route
                                CoreCoord forwarding_dest = forwarding_pairs[{route_chip_id, route_eth_physical_core}];
                                FabricRoutingLookup::ForwardingRoute forwarding_route = {
                                    route_eth_physical_core, forwarding_dest};
                                all_routes.push_back({route_chip_id, forwarding_route});
                            }
                        }
                        routing_path_map_[std::make_tuple(chip_id_src, src_router_eth_core, hop_count)] = all_routes;
                    }
                }
            }
        }

        // dump routing path map to screen
        // for (const auto& [key, value] : routing_path_map) {
        //    auto [chip_id_src, src_router_eth_core, hop_count] = key;
        //    log_info("Routing path map: chip_id:{}, router_x:{}, router_y:{}, hop_count:{}", chip_id_src,
        //    src_router_eth_core.x, src_router_eth_core.y, hop_count); for (const auto& [chip_id, route] : value) {
        //        log_info("  Chip ID: {}", chip_id);
        //        log_info("    Route: {},{} --> {},{}", route.first.x, route.first.y, route.second.x, route.second.y);
        //    }
        //}
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

    std::optional<FullFabricRoute> getRoutingPathsToDestination(
        chip_id_t chip_id,
        CoreCoord src_coord,
        CoreCoord first_eth_router_core_coord,
        int hops,
        CoreCoord dst_coord) const {
        auto it = routing_path_map_.find(std::make_tuple(chip_id, first_eth_router_core_coord, hops));
        if (it != routing_path_map_.end()) {
            FullFabricRoute route = it->second;
            if (route.size() >= 2) {
                // substitute src_coord and dst_coord as start and end points of route
                route.front().second.first = src_coord;
                route.back().second.second = dst_coord;
            }
            return route;
        }
        return std::nullopt;
    }

private:
    EthCoreToChannelMap eth_core_to_channel_lookup_;
    RoutingPathMap routing_path_map_;
};

}  // namespace tt_metal
}  // namespace tt
