// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_query_attention.hpp"

#include <xtensor/xnpy.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "modules/rotary_embedding.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"
namespace ttml::modules {

template <class E1, class E2>
std::pair<float, float> suggest_atol_rtol(
    std::string const& label, const E1& expected, const E2& actual, size_t first_n = 32) {
    auto abs_diffs = xt::abs(expected - actual);

    float max_abs_diff = xt::amax(abs_diffs)();
    float atol = max_abs_diff * 1.2f;  // 20 % safety margin

    constexpr float eps = 1e-5f;
    auto denom = xt::clip(xt::abs(expected), eps, std::numeric_limits<float>::max());
    auto rel_diffs = abs_diffs / denom;

    // ignore tiny expected values when taking the max
    auto valid_mask = xt::abs(expected) > eps;
    auto rel_pruned = xt::where(valid_mask, rel_diffs, 0.0f);  // zeros don't affect max
    float max_rel = xt::amax(rel_pruned)();
    float rtol = 1.2f * max_rel;

    fmt::println("[{}] suggested atol: {}", label, atol);
    fmt::println("[{}] suggested rtol: {}", label, rtol);

    auto expected_flat = xt::flatten(expected);
    auto actual_flat = xt::flatten(actual);
    size_t n = first_n == 0 ? expected_flat.size() : std::min(first_n, expected_flat.size());
    xt::xarray<float> expected_prefix = xt::view(expected_flat, xt::range(0, n));
    xt::xarray<float> actual_prefix = xt::view(actual_flat, xt::range(0, n));

    xt::xarray<float> zipped = xt::stack(xtuple(expected_prefix, actual_prefix), 1);
    fmt::println("[{}] expected vs actual (first {}): {}", label, n, zipped);

    fmt::println("[{}] expected shape: {}", label, expected.shape());
    fmt::println("[{}] actual shape: {}", label, actual.shape());
    return std::make_pair(atol, rtol);
}

GroupedQueryAttention::GroupedQueryAttention(const GQAConfig& config) :
    m_embedding_dim(config.embedding_dim), m_num_heads(config.num_heads), m_num_groups(config.num_groups) {
    // create layers
    m_q_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim, config.bias_linears);
    auto concat_kv_dim = 2U * m_num_groups * (m_embedding_dim / m_num_heads);
    m_kv_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, concat_kv_dim, config.bias_linears);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(config.dropout_prob);
    m_out_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim, config.bias_linears);
    m_embedding = std::make_shared<ttml::modules::RotaryEmbedding>(config.rope_params);

    // register modules
    create_name("grouped_query_attention");
    register_module(m_q_linear, "q_linear");
    register_module(m_kv_linear, "kv_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
    register_module(m_embedding, "embedding");
}

ttml::autograd::TensorPtr GroupedQueryAttention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto q = (*m_q_linear)(x);
    auto kv = (*m_kv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_heads, m_num_groups);

    if (m_embedding) {
        query_with_heads = (*m_embedding)(query_with_heads);
        key_with_heads = (*m_embedding)(key_with_heads);
    }

    xt::xarray<float> hf_query = xt::load_npy<float>("/home/j/intermediate_results/query_with_heads.npy");
    xt::xarray<float> hf_key = xt::load_npy<float>("/home/j/intermediate_results/key_with_heads.npy");
    xt::xarray<float> hf_value = xt::load_npy<float>("/home/j/intermediate_results/value_with_heads.npy");

    auto interleave_halves = [&](const xt::xarray<float>& x, int dim = -1) -> xt::xarray<float> {
        // Normalize dim to positive
        if (dim < 0) {
            dim += x.dimension();
        }

        size_t d = x.shape()[dim];
        assert(d % 2 == 0 && "hidden dim must be even");

        // Split the array along the specified dimension
        auto a = xt::view(x, xt::all(), xt::all(), xt::all(), xt::range(0, d / 2));
        auto b = xt::view(x, xt::all(), xt::all(), xt::all(), xt::range(d / 2, d));

        // Stack and reshape to get interleaved result
        auto stacked = xt::stack(xtuple(a, b), dim + 1);
        auto result_shape = x.shape();
        xt::xarray<float> reshaped = xt::reshape_view(stacked, result_shape);
        return reshaped;
    };

    hf_query = interleave_halves(hf_query);
    hf_key = interleave_halves(hf_key);

    xt::xarray<float> our_query = core::to_xtensor(query_with_heads->get_value());
    xt::xarray<float> our_key = core::to_xtensor(key_with_heads->get_value());
    xt::xarray<float> our_value = core::to_xtensor(value_with_heads->get_value());
    xt::dump_npy("/home/j/intermediate_results/our_value.npy", our_value);

    fmt::println("query shapes: {} (HF) vs {} (Our)", hf_query.shape(), our_query.shape());
    fmt::println("key shapes: {} (HF) vs {} (Our)", hf_key.shape(), our_key.shape());
    fmt::println("value shapes: {} (HF) vs {} (Our)", hf_value.shape(), our_value.shape());

    suggest_atol_rtol("query", hf_query, our_query);
    suggest_atol_rtol("key", hf_key, our_key);
    suggest_atol_rtol("value", hf_value, our_value);

    // Also print shapes for debugging
    fmt::println("Debug GQA - HF Query shape: {}", hf_query.shape());
    fmt::println("Debug GQA - Our Query shape: {}", our_query.shape());
    fmt::println("Debug GQA - HF Key shape: {}", hf_key.shape());
    fmt::println("Debug GQA - Our Key shape: {}", our_key.shape());
    fmt::println("Debug GQA - HF Value shape: {}", hf_value.shape());
    fmt::println("Debug GQA - Our Value shape: {}", our_value.shape());

    xt::xarray<float> hf_attn_output_raw = xt::load_npy<float>("/home/j/intermediate_results/attn_output_raw.npy");
    xt::xarray<float> hf_attn_output_fused = xt::load_npy<float>("/home/j/intermediate_results/attn_output_fused.npy");
    xt::xarray<float> hf_attn_output_projected =
        xt::load_npy<float>("/home/j/intermediate_results/attn_output_projected.npy");

    auto do_attn_steps_and_check = [&]() {
        auto query = query_with_heads;
        auto key = key_with_heads;
        auto value = value_with_heads;

        auto attention = ttml::ops::scaled_dot_product_attention(query, key, value, mask);

        xt::xarray<float> our_attn_output_raw = core::to_xtensor(attention->get_value());

        attention = ops::heads_fusion(attention);
        xt::xarray<float> our_attn_output_fused = core::to_xtensor(attention->get_value());
        auto out = (*m_out_linear)(attention);
        xt::xarray<float> our_attn_output_projected = core::to_xtensor(out->get_value());
        fmt::println(
            "Debug GQA - Attn Raw shape: {} (HF) vs {} (Our)", hf_attn_output_raw.shape(), our_attn_output_raw.shape());
        fmt::println(
            "Debug GQA - Attn Fused shape: {} (HF) vs {} (Our)",
            hf_attn_output_fused.shape(),
            our_attn_output_fused.shape());
        fmt::println(
            "Debug GQA - Attn Projected shape: {} (HF) vs {} (Our)",
            hf_attn_output_projected.shape(),
            our_attn_output_projected.shape());

        float max_diff_raw = xt::amax(xt::abs(hf_attn_output_raw - our_attn_output_raw))();
        float max_diff_fused = xt::amax(xt::abs(hf_attn_output_fused - our_attn_output_fused))();
        float max_diff_projected = xt::amax(xt::abs(hf_attn_output_projected - our_attn_output_projected))();

        fmt::println("Debug GQA - Max abs diff Raw: {}", max_diff_raw);
        fmt::println("Debug GQA - Max abs diff Fused: {}", max_diff_fused);
        fmt::println("Debug GQA - Max abs diff Projected: {}", max_diff_projected);

        out = (*m_dropout)(out);

        return out;
    };

    auto out = do_attn_steps_and_check();
    return out;
}

}  // namespace ttml::modules
