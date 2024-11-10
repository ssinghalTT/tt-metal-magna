// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

namespace tt::tt_metal {

struct SubDeviceId {
    using Id = uint8_t;
    Id id;

    Id to_index() const { return id; }

    SubDeviceId& operator++() {
        id++;
        return *this;
    }

    SubDeviceId operator++(int) {
        auto ret = *this;
        this->operator++();
        return ret;
    }

    SubDeviceId& operator+=(Id n) {
        id += n;
        return *this;
    }

    bool operator==(const SubDeviceId &other) const {
        return id == other.id;
    }

    auto operator <=>(const SubDeviceId &other) const {
        return id <=> other.id;
    }
};

struct SubDeviceManagerId {
    using Id = uint64_t;
    Id id;

    Id to_index() const { return id; }

    SubDeviceManagerId& operator++() {
        id++;
        return *this;
    }

    SubDeviceManagerId operator++(int) {
        auto ret = *this;
        this->operator++();
        return ret;
    }

    SubDeviceManagerId& operator+=(Id n) {
        id += n;
        return *this;
    }

    bool operator==(const SubDeviceManagerId &other) const {
        return id == other.id;
    }

    auto operator <=>(const SubDeviceManagerId &other) const {
        return id <=> other.id;
    }

};

}  // namespace tt::tt_metal


namespace std {

template <>
struct hash<tt::tt_metal::SubDeviceId> {
    std::size_t operator()(tt::tt_metal::SubDeviceId const &o) const {
        return std::hash<decltype(tt::tt_metal::SubDeviceId::id)>{}(o.to_index());
    }
};

template <>
struct hash<tt::tt_metal::SubDeviceManagerId> {
    std::size_t operator()(tt::tt_metal::SubDeviceManagerId const &o) const {
        return std::hash<decltype(tt::tt_metal::SubDeviceManagerId::id)>{}(o.to_index());
    }
};

}  // namespace std
