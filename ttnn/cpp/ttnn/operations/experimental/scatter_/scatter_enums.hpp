// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <type_traits>

#include "hostdevcommon/kernel_structs.h"

namespace ttnn::operations::experimental::scatter {

enum class ScatterReductionType : uint8_t { ADD, MULTIPLY };

enum class ScatterCB : std::underlying_type_t<tt::CBIndex> { INPUT, INDEX, DST };

}  // namespace ttnn::operations::experimental::scatter
