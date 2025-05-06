// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/assert.hpp>

namespace ttnn::operations::pool {
// Return a single bf16 scalar for the pool type in u32 (packed in the least 16 bits)
std::vector<uint32_t> get_bf16_pool_scalar(
    Pool2DType pool_type,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t kernel_size_h,
    uint32_t kernel_size_w,
    bool ceil_mode,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t ceil_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t out_stick_x_start,
    uint32_t out_stick_y_start,
    std::optional<uint32_t> out_nhw_per_core,
    std::optional<int32_t> divisor_override) {
    std::vector<uint32_t> scalars;
    float value;
    float previous_value = 0.;
    bool includes_padding = false;
    bool reached_padding = false;
    bool first = true;
    bool should_wait_for_new_scalar = false;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D:
            value = 1.;
            scalars.push_back(bfloat16(value).to_packed());
            break;
        case Pool2DType::AVG_POOL2D:
            if (divisor_override.has_value()) {
                value = 1. / (float)divisor_override.value();
                scalars.push_back(bfloat16(value).to_packed());
            } else if (ceil_mode) {
                for (uint32_t i = 0; i < out_nhw_per_core.value(); i++) {
                    value = 1. / (float)(kernel_size_h * kernel_size_w);
                    if ((in_w - kernel_size_w) % stride_w != 0) {
                        if (out_stick_x_start == out_w - 1 && out_stick_y_start == out_h - 1) {
                            value = 1. / (float)((kernel_size_h * kernel_size_w) - kernel_size_h * ceil_w -
                                                 (kernel_size_w - ceil_w) * ceil_w);
                        } else if (out_stick_x_start == out_w - 1) {
                            value = 1. / (float)((kernel_size_h * kernel_size_w) - kernel_size_w * ceil_w);
                        } else if (out_stick_y_start == out_h - 1) {
                            value = 1. / (float)((kernel_size_h * kernel_size_w) - kernel_size_w * ceil_w);
                        }
                    }
                    bool includes_padding = (out_stick_y_start * stride_h - pad_h) < 0 ||
                                            (out_stick_y_start * stride_h - pad_h + kernel_size_h) > (in_h + ceil_w) ||
                                            (out_stick_x_start * stride_w - pad_w) < 0 ||
                                            (out_stick_x_start * stride_w - pad_w + kernel_size_w) > (in_w + ceil_w);

                    if ((reached_padding && !includes_padding) || includes_padding || first) {
                        first = false;
                        should_wait_for_new_scalar = true;
                    }
                    if (includes_padding) {
                        reached_padding = true;
                    }

                    if (should_wait_for_new_scalar) {
                        scalars.push_back(bfloat16(value).to_packed());
                    }

                    out_stick_y_start = (out_stick_y_start + 1) % out_h;
                    if (out_stick_y_start == 0) {
                        out_stick_x_start = (out_stick_x_start + 1) % out_w;
                    }
                }

            } else {
                value = 1. / (float)(kernel_size_h * kernel_size_w);
                scalars.push_back(bfloat16(value).to_packed());
            }
            break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }

    return scalars;
}

// Return a single bf16 init value for the pool type in u32 (packed in the least 16 bits)
uint32_t get_bf16_pool_init_value(Pool2DType pool_type) {
    float value;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = -std::numeric_limits<float>::infinity(); break;
        case Pool2DType::AVG_POOL2D: value = 0.; break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    return bfloat16(value).to_packed();
}

std::map<std::string, std::string> get_defines(Pool2DType pool_type) {
    std::map<std::string, std::string> defines;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: defines["REDUCE_OP"] = "PoolType::MAX"; break;
        case Pool2DType::AVG_POOL2D: defines["REDUCE_OP"] = "PoolType::SUM"; break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";

    return defines;
}
}  // namespace ttnn::operations::pool
