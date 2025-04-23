// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "models/llama.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <xtensor/xnpy.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "serialization/serialization.hpp"

class LlamaTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(LlamaTest, RopeFreqs) {
    using namespace ttml;
    auto seq_len = 32;
    auto head_dim = 64;
    auto theta = 10000.0F;
    auto rope_params = ttml::ops::build_rope_params(seq_len, head_dim, theta);
    // xt::xarray<float> expected_cos_freqs = xt::load_npy<float>("/home/ubuntu/intermediate_results/cos_freqs.npy");
    // xt::xarray<float> expected_sin_freqs = xt::load_npy<float>("/home/ubuntu/intermediate_results/sin_freqs.npy");

    // fmt::println("cos freqs shape: {}", expected_cos_freqs.shape());
    // fmt::println("sin freqs shape: {}", expected_sin_freqs.shape());

    // auto actual_cos_freqs = core::to_xtensor(rope_params.cos_cache);
    // auto actual_sin_freqs = core::to_xtensor(rope_params.sin_cache);

    // actual_cos_freqs = actual_cos_freqs.reshape({1, seq_len, head_dim});
    // actual_sin_freqs = actual_sin_freqs.reshape({1, seq_len, head_dim});

    // auto average_diff_cos = xt::mean(xt::abs(actual_cos_freqs - expected_cos_freqs))();
    // auto average_diff_sin = xt::mean(xt::abs(actual_sin_freqs - expected_sin_freqs))();

    // fmt::println("average diff for cos: {}", average_diff_cos);
    // fmt::println("average diff for sin: {}", average_diff_sin);

    // fmt::println("First 64 expected cos freqs:");
    // for (size_t i = 0; i < 64 && i < expected_cos_freqs.shape()[2]; ++i) {
    //     fmt::print("{} ", expected_cos_freqs(0, 1, i));
    // }
    // fmt::println("");

    // fmt::println("First 64 actual cos freqs:");
    // for (size_t i = 0; i < 64 && i < actual_cos_freqs.shape()[2]; ++i) {
    //     fmt::print("{} ", actual_cos_freqs(0, 1, i));
    // }
    // fmt::println("");

    // fmt::println("First 64 expected sin freqs:");
    // for (size_t i = 0; i < 64 && i < expected_sin_freqs.shape()[2]; ++i) {
    //     fmt::print("{} ", expected_sin_freqs(0, 1, i));
    // }
    // fmt::println("");

    // fmt::println("First 64 actual sin freqs:");
    // for (size_t i = 0; i < 64 && i < actual_sin_freqs.shape()[2]; ++i) {
    //     fmt::print("{} ", actual_sin_freqs(0, 1, i));
    // }
    // fmt::println("");

    // EXPECT_TRUE(average_diff_cos < 2e-2F);
    // EXPECT_TRUE(average_diff_sin < 2e-2F);
    fmt::println("testing overall rope");
    xt::xarray<float> rope_input_xt =
        xt::load_npy<float>("/home/ubuntu/intermediate_results/query_states_before_rope.npy");
    auto interleave_halves = [&](const xt::xarray<float>& x, int dim = -1) {
        fmt::println("interleave halves 1");
        // Normalize dim to positive
        if (dim < 0) {
            dim += x.dimension();
        }
        fmt::println("interleave halves 2: dim = {}", dim);
        fmt::println("interleave halves 3: input shape = {}", x.shape());

        size_t d = x.shape()[dim];
        assert(d % 2 == 0 && "hidden dim must be even");

        // Split the array along the specified dimension
        auto a = xt::view(x, xt::all(), xt::all(), xt::all(), xt::range(0, d / 2));
        auto b = xt::view(x, xt::all(), xt::all(), xt::all(), xt::range(d / 2, d));
        fmt::println("interleave halves 4: shape a = {}, shape b = {}", a.shape(), b.shape());

        // Stack and reshape to get interleaved result
        auto stacked = xt::stack(xtuple(a, b), dim + 1);
        fmt::println("interleave halves 5: stacked shape = {}", stacked.shape());
        auto result_shape = x.shape();
        fmt::println("interleave halves 6: result shape = {}", result_shape);
        xt::xarray<float> reshaped = xt::reshape_view(stacked, result_shape);
        fmt::println("interleave halves 7: reshaped shape = {}", reshaped.shape());
        return reshaped;
    };
    fmt::println("interleaving rope input");
    rope_input_xt = interleave_halves(rope_input_xt, -1);
    fmt::println("interleaved rope input.");
    fmt::println("rope input shape: {}", rope_input_xt.shape());
    rope_input_xt = rope_input_xt.reshape({1, 32, 32, head_dim});
    auto rope_input = autograd::create_tensor(core::from_xtensor(rope_input_xt, &autograd::ctx().get_device()));
    auto rope_res = ttml::ops::rope(rope_input, rope_params);
    auto rope_res_xt = core::to_xtensor(rope_res->get_value());
    fmt::println("rope res shape: {}", rope_res_xt.shape());
    rope_res_xt = rope_res_xt.reshape({1, 32, 32, head_dim});
    auto expected_rope_res = xt::load_npy<float>("/home/ubuntu/intermediate_results/query_states_after_rope.npy");
    expected_rope_res = interleave_halves(expected_rope_res, -1);
    auto average_diff_rope = xt::mean(xt::abs(expected_rope_res - rope_res_xt))();
    fmt::println("average diff for rope: {}", average_diff_rope);

    for (size_t tok = 0; tok < 32; ++tok) {
        fmt::print("rope act: ");
        for (size_t i = 32; i < 64; ++i) {
            fmt::print("{:8.4f} ", rope_res_xt(0, 0, tok, i));
        }
        fmt::print("\n");

        fmt::print("rope exp: ");
        for (size_t i = 32; i < 64; ++i) {
            fmt::print("{:8.4f} ", expected_rope_res(0, 0, tok, i));
        }
        fmt::print("\n");
    }
}

TEST_F(LlamaTest, ForwardPhases) {
    using namespace ttml;
    xt::xarray<float> input_ids_xt = xt::load_npy<float>("/home/ubuntu/intermediate_results/test_input_tokens.npy");
    xt::xarray<float> attention_mask_xt =
        xt::load_npy<float>("/home/ubuntu/intermediate_results/test_attention_mask.npy");

    auto x = autograd::create_tensor(core::from_xtensor(input_ids_xt, &autograd::ctx().get_device()));
    auto mask = autograd::create_tensor(core::from_xtensor(attention_mask_xt, &autograd::ctx().get_device()));

    fmt::println("current working directory: {}", std::filesystem::current_path());
    auto yaml_config = YAML::LoadFile("configs/training_shakespeare_llama3.yaml");
    auto training_config = yaml_config["training_config"];
    auto llama_config = training_config["transformer_config"];
    auto config = models::llama::read_config(llama_config);
    config.max_sequence_length = 32;

    auto llama_model = models::llama::Llama(config);
    llama_model.eval();

    fmt::println("loading tinyllama.msgpack");
    ttml::serialization::MsgPackFile tinyllama_msgpack{};
    tinyllama_msgpack.deserialize("/home/ubuntu/tinyllama.msgpack");
    fmt::println("deserialized tinyllama.msgpack");

    /* load weights using ttml::serialization::read_module*/
    fmt::println("loading weights");
    ttml::serialization::read_module(tinyllama_msgpack, /*name=*/"llama", &llama_model);
    fmt::println("loaded weights");

    auto tok_emb = llama_model.tok_emb;
    auto tok_emb_res = (*tok_emb)(x);

    fmt::println("checking tok_emb_res");
    auto actual_tok_emb_res = core::to_xtensor(tok_emb_res->get_value());
    auto expected_tok_emb_res = xt::load_npy<float>("/home/ubuntu/intermediate_results/embedded.npy");
    auto average_diff = xt::mean(xt::abs(expected_tok_emb_res - actual_tok_emb_res))();
    fmt::println("average diff for emb: {}", average_diff);

    fmt::println("expected tok emb shape: {}", expected_tok_emb_res.shape());
    fmt::println("actual tok emb shape: {}", actual_tok_emb_res.shape());

    EXPECT_TRUE(average_diff < 2e-2F);

    xt::xarray<float> expected_first_block_res =
        xt::load_npy<float>("/home/ubuntu/intermediate_results/expected_first_block_output.npy");
    auto first_block = llama_model.blocks[0];
    auto actual_first_block_res = core::to_xtensor((*first_block)(tok_emb_res, mask)->get_value());
    // reshape both to 1,32,32,32
    std::vector<uint32_t> actual_shape(actual_first_block_res.shape().begin(), actual_first_block_res.shape().end());
    std::vector<uint32_t> expected_shape(
        expected_first_block_res.shape().begin(), expected_first_block_res.shape().end());
    fmt::println("actual shape: {}", actual_shape);
    fmt::println("expected shape: {}", expected_shape);
    actual_first_block_res = actual_first_block_res.reshape({1U, 32U, 2048U});
    auto average_diff_blocks = xt::mean(xt::abs(expected_first_block_res - actual_first_block_res))();
    fmt::println("average diff for first block: {}", average_diff_blocks);
    EXPECT_TRUE(average_diff_blocks < 2e-2F);

    fmt::println("checking the mlp");
    std::shared_ptr<ttml::modules::LlamaBlock> first_block_ptr =
        std::dynamic_pointer_cast<ttml::modules::LlamaBlock>(first_block);
    auto mlp = first_block_ptr->m_mlp;

    xt::xarray<float> mlp_input_xt = xt::load_npy<float>("/home/ubuntu/intermediate_results/mlp_test_input.npy");
    auto mlp_input = autograd::create_tensor(core::from_xtensor(mlp_input_xt, &autograd::ctx().get_device()));
    auto mlp_res = (*mlp)(mlp_input);
    auto mlp_res_xt = core::to_xtensor(mlp_res->get_value());
    fmt::println("mlp res shape: {}", mlp_res_xt.shape());
    auto expected_mlp_res = xt::load_npy<float>("/home/ubuntu/intermediate_results/mlp_test_output.npy");
    expected_mlp_res = expected_mlp_res.reshape({2048});
    mlp_res_xt = mlp_res_xt.reshape({2048});
    fmt::println("expected mlp res shape: {}", expected_mlp_res.shape());
    auto average_diff_mlp = xt::mean(xt::abs(expected_mlp_res - mlp_res_xt))();
    fmt::println("average diff for mlp: {}", average_diff_mlp);

    fmt::println("MLP diffs: {}", xt::xarray<float>(xt::abs(expected_mlp_res - mlp_res_xt)));

    fmt::println("mlp first tile: ");
    for (size_t i = 0; i < 32; ++i) {
        fmt::print("{:8.4f} ", mlp_res_xt(i));
    }
    fmt::print("\n");

    fmt::println("expected mlp first tile: ");
    for (size_t i = 0; i < 32; ++i) {
        fmt::print("{:8.4f} ", expected_mlp_res(i));
    }
    fmt::print("\n");

    EXPECT_TRUE(average_diff_mlp < 2e-2F);

    // now checking the self attention module
    auto attention_norm = first_block_ptr->m_attention_norm;
    auto attention = first_block_ptr->m_attention;
    auto attention_res = (*attention_norm)(tok_emb_res);
    attention_res = (*attention)(attention_res, mask);
    auto attention_res_xt = core::to_xtensor(attention_res->get_value());
    auto expected_attention_res = xt::load_npy<float>("/home/ubuntu/intermediate_results/self_attn_output.npy");
    auto average_diff_attention = xt::mean(xt::abs(expected_attention_res - attention_res_xt))();
    fmt::println("average diff for attention: {}", average_diff_attention);

    fmt::println("attention shape: {}", attention_res_xt.shape());
    attention_res_xt = attention_res_xt.reshape({1, 32, 2048});
    fmt::println("expected attention shape: {}", expected_attention_res.shape());

    fmt::println("self attention first tile diagonal: ");
    for (size_t i = 0; i < std::min(attention_res_xt.shape()[1], attention_res_xt.shape()[2]); ++i) {
        fmt::print("{:8.4f} ", attention_res_xt(0, i, i));
    }
    fmt::print("\n");

    fmt::println("expected self attention first tile diagonal: ");
    for (size_t i = 0; i < std::min(expected_attention_res.shape()[1], expected_attention_res.shape()[2]); ++i) {
        fmt::print("{:8.4f} ", expected_attention_res(0, i, i));
    }
    fmt::print("\n");

    EXPECT_TRUE(average_diff_attention < 2e-2F);
}
