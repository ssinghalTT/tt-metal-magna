// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_op.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/run_operation.hpp"

#include "pad.hpp"

namespace ttnn::operations::data_movement {

namespace {

bool eq_spans(const auto a, const auto b) { return std::equal(a.begin(), a.end(), b.begin(), b.end()); }

ttnn::Shape update_original_shape(const ttnn::Shape& padded_shape, const ttnn::Shape& input_shape) {
    ttnn::SmallVector<uint32_t> updated_shape;
    size_t input_rank = input_shape.rank();
    for (size_t i = 0; i < input_rank - 2; i++) {
        updated_shape.push_back(input_shape[i]);
    }
    updated_shape.push_back(padded_shape[-2]);
    updated_shape.push_back(padded_shape[-1]);
    return ttnn::Shape(std::move(updated_shape));
}

static ttnn::Tensor pad_impl(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    std::span<const uint32_t> output_padded_shape,
    std::span<const uint32_t> input_tensor_start,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {
    auto input_logical_shape = input_tensor.logical_shape().view();
    // on host
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        if (eq_spans(input_logical_shape, output_padded_shape)) {
            return input_tensor;
        } else {
            return input_tensor.pad(ttnn::Shape(output_padded_shape), ttnn::Shape{input_tensor_start}, value);
        }
    }

    // on device
    else {
        auto input_tensor_shape = input_tensor.get_logical_shape();
        const auto rank = input_tensor_shape.rank();

        TT_FATAL(rank == 4, "ttnn.pad: input tensor passed to pad_impl must have rank == 4, but got rank {}.", rank);
        bool input_output_same = true;
        for (size_t i = 0; i < rank; i++) {
            if (input_tensor_shape[i] != output_padded_shape[i]) {
                input_output_same = false;
                break;
            }
        }
        if (input_output_same) {
            tt::log_debug("Pad Input and Output Shapes are the same. Skipping pad and returning input tensor.");
            return input_tensor;
        }
        using ShardStrategy = ttnn::operations::data_movement::ShardStrategy;
        using ShardOrientation = tt::tt_metal::ShardOrientation;
        using Layout = tt::tt_metal::Layout;

        auto output_memory_config = memory_config_arg.value_or(input_tensor.memory_config());

        if (input_tensor.is_sharded()) {
            auto total_height = [](const auto& shape) {
                return std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<uint32_t>());
            };

            auto height_distinct = [&total_height](const auto& shape, const auto& other_shape) {
                return total_height(shape) != total_height(other_shape);
            };

            auto width_distinct = [](const auto& shape, const auto& other_shape) { return shape[3] != other_shape[3]; };

            uint32_t input_w = input_logical_shape[3];
            uint32_t output_w = output_padded_shape[3];

            if (width_distinct(input_logical_shape, output_padded_shape)) {
                std::array<uint32_t, 4> output_shape_width_padded{
                    input_logical_shape[0], input_logical_shape[1], input_logical_shape[2], output_w};
                auto width_pad_memory_config = create_sharded_memory_config(
                    ttnn::Shape{output_shape_width_padded},
                    input_tensor.shard_spec()->grid,  // reuse input cores for now: FIXME: can we do better?
                                                      // it's complicated because we need the input shards to be local
                                                      // to the core holding the output shard currently.
                    ShardStrategy::HEIGHT,            // stay height sharded
                    ShardOrientation::ROW_MAJOR);
                output_memory_config = width_pad_memory_config;

                if (height_distinct(input_logical_shape, output_padded_shape)) {
                    // we will decompose the padding into two parts and run two
                    // separate pads.
                    ttnn::SmallVector<uint32_t> adjusted_input_tensor_start{0, 0, 0, input_tensor_start[3]};

                    TT_ASSERT(
                        not(height_distinct(input_logical_shape, output_shape_width_padded) and
                            width_distinct(input_logical_shape, output_shape_width_padded)),
                        "infinite recursion");

                    // pad width
                    auto output_tensor_width_padded = pad_impl(
                        queue_id,
                        input_tensor,
                        output_shape_width_padded,
                        adjusted_input_tensor_start,
                        value,
                        use_multicore,
                        width_pad_memory_config);

                    TT_ASSERT(
                        not(height_distinct(output_padded_shape, output_shape_width_padded) and
                            width_distinct(output_padded_shape, output_shape_width_padded)),
                        "infinite recursion");

                    auto height_pad_memory_config = create_sharded_memory_config(
                        ttnn::Shape{output_padded_shape},
                        input_tensor.shard_spec()->grid,
                        ShardStrategy::HEIGHT,
                        ShardOrientation::ROW_MAJOR);

                    // then pad height
                    auto output_tensor_height_padded = pad_impl(
                        queue_id,
                        output_tensor_width_padded,
                        output_padded_shape,
                        input_tensor_start,
                        value,
                        use_multicore,
                        memory_config_arg.value_or(height_pad_memory_config));
                    output_tensor_width_padded.deallocate();  // dealloc temporary width padded tensor
                    return output_tensor_height_padded;
                }
            }
        }

        auto output_w = output_padded_shape[3];
        TT_ASSERT(
            !input_tensor.is_sharded() || output_w == output_memory_config.shard_spec->shape[1],
            "output_w != output_memory_config.shard_spec().shape[1]");

        ttnn::Shape output_shape{output_padded_shape};
        auto output_tensor = tt::tt_metal::operation::run(
                                 Pad{output_shape,
                                     output_shape,
                                     ttnn::Shape{input_tensor_start},
                                     value,
                                     output_memory_config,
                                     use_multicore},
                                 {input_tensor},
                                 {},
                                 {},
                                 queue_id)
                                 .front();

        return output_tensor;
    }
}

static ttnn::Tensor pad_impl(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    ttnn::SmallVector<std::pair<uint32_t, uint32_t>> padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {
    const int original_rank = input_tensor.get_logical_shape().rank();

    TT_FATAL(padding.size() == original_rank, "ttnn.pad: padding must be the same length as the input tensor rank");

    // Unsqueeze Tensor to 4D if it is not already
    ttnn::Tensor input_tensor_4D;
    if (input_tensor.get_logical_shape().rank() < 4) {
        input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    } else if (input_tensor.get_logical_shape().rank() > 4) {
        input_tensor_4D = squeeze_from_ND_to_4D(input_tensor);
    } else {
        input_tensor_4D = input_tensor;
    }
    size_t padding_size = 4;
    size_t extra_index = input_tensor.get_logical_shape().rank() - 4;
    if (input_tensor.get_logical_shape().rank() < 4) {
        padding.insert(padding.begin(), 4 - original_rank, {0, 0});
        padding_size = padding.size();
        extra_index = 0;
    }
    auto input_shape_with_tile_padding = input_tensor_4D.get_padded_shape();
    std::vector<uint32_t> output_padded_shape(padding_size, 0);
    for (size_t i = 0; i < padding_size; i++) {
        output_padded_shape[i] =
            padding[i + extra_index].first + input_shape_with_tile_padding[i] + padding[i + extra_index].second;
    }

    auto pad_front = padding | std::views::transform([](const auto& p) { return p.first; });
    auto pad_back = padding | std::views::transform([](const auto& p) { return p.second; });

    if (input_tensor.get_layout() == ttnn::TILE_LAYOUT) {
        const int target_height = output_padded_shape[padding_size - 2];
        const int target_width = output_padded_shape[padding_size - 1];
        TT_FATAL(
            target_height % ttnn::TILE_SIZE == 0 || target_width % ttnn::TILE_SIZE == 0,
            "ttnn.pad: for tiled tensors padding end must be a multiple of the tile size on height and width for a "
            "tensor in tile layout");
    }

    // Performing actual padding
    std::vector<uint32_t> pad_front_array(padding_size, 0);
    for (size_t i = 0; i < pad_front.size(); i++) {
        pad_front_array[i] = pad_front[i];
    }

    return pad_impl(
        queue_id, input_tensor_4D, output_padded_shape, pad_front_array, value, use_multicore, memory_config_arg);
}

std::tuple<ttnn::Shape, ttnn::Shape> compute_requested_shape(
    const ttnn::Shape& input_logical_shape, const ttnn::SmallVector<std::pair<uint32_t, uint32_t>>& pad_spec) {
    if (std::all_of(pad_spec.begin(), pad_spec.end(), [](auto& p) { return p.first == 0 && p.second == 0; })) {
        return std::make_tuple(compute_padded_shape(input_logical_shape), compute_padded_shape(input_logical_shape));
    }

    const auto rank = input_logical_shape.rank();
    ttnn::SmallVector<uint32_t> requested_logical_shape_vec(rank, 0);

    std::transform(
        input_logical_shape.cbegin(),
        input_logical_shape.cend(),
        pad_spec.cbegin(),
        requested_logical_shape_vec.begin(),
        [](auto& a, auto& b) { return a + b.second; });

    const ttnn::Shape logical_shape(requested_logical_shape_vec);
    return std::make_tuple(logical_shape, compute_padded_shape(logical_shape));
}

ttnn::Tensor invoke_rm(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<std::pair<uint32_t, uint32_t>>& padding_vec,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {
    const int original_rank = input_tensor.get_logical_shape().rank();

    ttnn::Tensor output_tensor =
        pad_impl(queue_id, input_tensor, std::move(padding_vec), value, use_multicore, memory_config_arg);

    // output_tensor is currently 4D. We have to squeeze back to the original rank
    if (original_rank <= 4) {
        auto to_vec = [](const auto& span) { return ttnn::SmallVector<uint32_t>{span.begin(), span.end()}; };
        auto output_shape = to_vec(output_tensor.get_padded_shape().view());
        auto padded_shape = to_vec(output_tensor.get_padded_shape().view());
        if (const auto rank_diff = output_shape.size() - original_rank; rank_diff) {
            auto remove_prefix = [](auto& source, size_t n) { source.erase(source.begin(), source.begin() + n); };
            remove_prefix(output_shape, rank_diff);
            remove_prefix(padded_shape, rank_diff);
            output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(output_shape), ttnn::Shape(padded_shape));
            output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(padded_shape));
        }
    } else {
        output_tensor = ttnn::reshape(
            output_tensor, update_original_shape(output_tensor.get_padded_shape(), input_tensor.get_logical_shape()));
    }
    return output_tensor;
}

ttnn::Tensor invoke_tile(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<std::pair<uint32_t, uint32_t>>& padding_vec,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {
    const bool front_padding_is_zero =
        std::all_of(padding_vec.begin(), padding_vec.end(), [](auto& x) { return x.first == 0; });
    TT_FATAL(front_padding_is_zero, "ttnn.pad: on device tile padding does not support front padding");

    const auto& input_logical_shape = input_tensor.get_logical_shape();
    const auto& input_padded_shape = input_tensor.get_padded_shape();
    const auto [requested_logical_shape, requested_padded_shape] =
        compute_requested_shape(input_logical_shape, padding_vec);
    const auto requested_rank = requested_logical_shape.rank();

    const bool pad_upper_dims =
        requested_logical_shape[0] != input_logical_shape[0] || requested_logical_shape[1] != input_logical_shape[1];

    auto pad_current_tile_dim = [&requested_padded_shape, &input_logical_shape](const int i) {
        return requested_padded_shape[i] / input_logical_shape[i] == 1;
    };

    ttnn::Tensor output_tensor = ttnn::fill_implicit_tile_padding(input_tensor, value, memory_config_arg);
    if (requested_rank == 1 || (!pad_upper_dims && pad_current_tile_dim(-1) && pad_current_tile_dim(-2))) {
        output_tensor = ttnn::experimental::view(output_tensor, requested_logical_shape, requested_padded_shape);
    } else {
        // need to align the requested padding to tile size. Note that begin padding is not supported so now just
        // set to zero
        ttnn::SmallVector<std::pair<uint32_t, uint32_t>> padded_padding_vec(requested_rank);
        std::transform(
            requested_padded_shape.cbegin(),
            requested_padded_shape.cend(),
            input_padded_shape.cbegin(),
            padded_padding_vec.begin(),
            [](auto& a, auto& b) { return std::make_pair(0, a - b); });

        // this tensor will be 4D
        output_tensor =
            pad_impl(queue_id, output_tensor, std::move(padded_padding_vec), value, use_multicore, memory_config_arg);

        // this is the padded shape
        const auto output_shape = squeeze_or_unsqueeze_shape_to_ND(output_tensor.get_logical_shape(), requested_rank);

        // "slice" down to logical shape
        output_tensor = ttnn::experimental::view(output_tensor, requested_logical_shape, requested_padded_shape);
    }
    return output_tensor;
}
}  // anonymous namespace

// This function signature is similar to pytorch's signature
// Any rank tensor supported

ttnn::Tensor ExecutePad::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const std::pair<uint32_t, uint32_t>> padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {
    const int original_rank = input_tensor.get_logical_shape().rank();
    ttnn::SmallVector<std::pair<uint32_t, uint32_t>> padding_vec(padding.begin(), padding.end());

    if (int diff = original_rank - padding.size(); diff != 0) {
        TT_FATAL(diff > 0, "ttnn.pad: padding len can't be larger than input tensor rank");

        padding_vec.insert(padding_vec.begin(), diff, {0, 0});
    }

    if (std::all_of(padding.begin(), padding.end(), [](auto& p) { return p.first == 0 && p.second == 0; })) {
        return input_tensor;
    }

    if (original_rank > 4) {
        const auto first_pad_idx =
            std::find_if(padding.begin(), padding.end(), [](auto& p) { return p.second != 0; }) - padding.begin();
        TT_FATAL(
            first_pad_idx >= original_rank - 3,
            "ttnn::pad only supports padding on the lowest 3 dimensions for tensors with rank > 4 {}",
            first_pad_idx);
    }

    if (input_tensor.get_layout() == ttnn::TILE_LAYOUT) {
        return invoke_tile(queue_id, input_tensor, padding_vec, value, use_multicore, memory_config_arg);
    } else {
        return invoke_rm(queue_id, input_tensor, padding_vec, value, use_multicore, memory_config_arg);
    }
}

#define PAD_OVERLOAD_DIM_IMPL(ShapeType)                                                                               \
    ttnn::Tensor ExecutePad::invoke(                                                                                   \
        QueueId queue_id,                                                                                              \
        const ttnn::Tensor& input_tensor,                                                                              \
        const ShapeType& output_padded_shape,                                                                          \
        const ShapeType& input_tensor_start,                                                                           \
        const float value,                                                                                             \
        const bool use_multicore,                                                                                      \
        const std::optional<MemoryConfig>& memory_config_arg) {                                                        \
        return pad_impl(                                                                                               \
            queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config_arg); \
    }                                                                                                                  \
                                                                                                                       \
    ttnn::Tensor ExecutePad::invoke(                                                                                   \
        const ttnn::Tensor& input_tensor,                                                                              \
        const ShapeType& output_padded_shape,                                                                          \
        const ShapeType& input_tensor_start,                                                                           \
        const float value,                                                                                             \
        const std::optional<MemoryConfig>& memory_config_arg) {                                                        \
        return pad_impl(                                                                                               \
            DefaultQueueId, input_tensor, output_padded_shape, input_tensor_start, value, false, memory_config_arg);   \
    }                                                                                                                  \
                                                                                                                       \
    ttnn::Tensor ExecutePad::invoke(                                                                                   \
        const ttnn::Tensor& input_tensor,                                                                              \
        const ShapeType& output_padded_shape,                                                                          \
        const ShapeType& input_tensor_start,                                                                           \
        const float value) {                                                                                           \
        return pad_impl(                                                                                               \
            DefaultQueueId, input_tensor, output_padded_shape, input_tensor_start, value, false, std::nullopt);        \
    }

PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array1D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array2D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array3D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array4D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array5D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array6D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array7D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array8D)

}  // namespace ttnn::operations::data_movement
