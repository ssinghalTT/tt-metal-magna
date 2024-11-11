# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import transformers
import torch
import math
from ttnn.model_preprocessing import (
    preprocess_linear_weight,
    preprocess_linear_bias,
)

import ttnn
from ttnn.dot_access import DotAccessDict


def update_model_config(config, batch_size):
    core_grid = ttnn.CoreGrid(y=8, x=8)
    seqL_t = 4096 // 32 // core_grid.y  # 7
    dim_t = 768 // 32  # 24
    dim_t__x = dim_t // core_grid.x  # 4
    head_num = 12
    head_seqL_t = 4096 // 32  # 7
    head_size_t__x = dim_t // head_num  # 2
    class__x = (1152 // 32) // core_grid.x  # 3
    slice_ratio = 1

    # sharding configs
    program_configs = {
        "query_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t__x,  # 2,
            out_subblock_h=1,
            out_subblock_w=int(2 * dim_t__x / 2),  # 6,
            per_core_M=seqL_t,  # 7,
            per_core_N=dim_t__x,  # 12,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "query_key_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t__x // 2,  # 2,
            out_subblock_h=1,
            out_subblock_w=dim_t__x,  # 6,
            per_core_M=seqL_t,  # 7,
            per_core_N=3 * dim_t__x,  # 12,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "query_by_key_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=(4096 // 32 // (core_grid.y * core_grid.x)),  # 2,
            out_subblock_h=1,
            out_subblock_w=(4096 // 32),  # 7,
            per_core_M=int((4096 // 32 // (core_grid.y * core_grid.x)) * slice_ratio),  # 14,
            per_core_N=(4096 // 32),  # 7,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        "attention_probabilities_by_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=(4096 // 32),  # 2,
            out_subblock_h=1,
            out_subblock_w=2,  # 7,
            per_core_M=(4096 // 32 // (core_grid.y * core_grid.x)),  # 14,
            per_core_N=2,  # 7,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        "self_output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=int(dim_t__x / 2),  # 2,
            out_subblock_h=1,  # seqL_t,  # 7,
            out_subblock_w=dim_t__x,  # 4,
            per_core_M=seqL_t,  # 7,
            per_core_N=dim_t__x,  # 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=int(dim_t__x / 2),  # 2,
            out_subblock_h=1,
            out_subblock_w=int(4 * dim_t__x / 4),  # 4,
            per_core_M=seqL_t,  # 7,
            per_core_N=int(4 * dim_t__x),  # 16,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=int(4 * dim_t__x / 2),  # 8,
            out_subblock_h=1,  # seqL_t,  # 7,
            out_subblock_w=dim_t__x,  # 4,
            per_core_M=seqL_t,  # 7,
            per_core_N=dim_t__x,  # 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "classifer_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=int(dim_t__x / 2),  # 2,
            out_subblock_h=1,
            out_subblock_w=class__x,  # 3,
            per_core_M=seqL_t,  # 7,
            per_core_N=class__x,  # 6,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "layernorm_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=int(dim_t__x / 2),  # 2,
            block_h=seqL_t,  # 7,
            block_w=dim_t__x,  # 4,
            inplace=False,
        ),
        "layernorm_after_output_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=int(dim_t__x / 2),  # 2,
            block_h=seqL_t,  # 7,
            block_w=dim_t__x,  # 4,
            inplace=False,
        ),
        "softmax_program_config": ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=(4096 // 32),  # 7,
            block_h=(4096 // 32 // (core_grid.y * core_grid.x)),  # 14,
            block_w=(4096 // 32),  # 7,
        ),
    }

    return DotAccessDict(dict(**config.to_dict(), core_grid=core_grid, program_configs=program_configs))


# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


def vit_patch_embeddings(config, pixel_values, *, parameters, unittest_check=False):
    # batch_size, img_c, img_h, img_w = pixel_values.shape # NCHW
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = 16
    patch_count = img_h // patch_size  # 14
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    folded_pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)  # 1568, 1024
    ttnn.deallocate(pixel_values)
    folded_pixel_values = ttnn.to_memory_config(folded_pixel_values, memory_config=ttnn.L1_MEMORY_CONFIG)
    # Convert back to interleaved or otherwise to_layout will fail
    folded_pixel_values = ttnn.to_layout(folded_pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn.linear(
        folded_pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=config.core_grid,
    )

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, patch_size_sq_trpl))

    return patch_embedding_output


def vit_embeddings(
    config,
    pixel_values,
    cls_token,
    position_embeddings,
    *,
    parameters,
):
    parameters = parameters.vit.embeddings

    l1_memory_config = ttnn.L1_MEMORY_CONFIG

    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    embedding_output = ttnn.concat([cls_token, patch_embeddings], -2, memory_config=l1_memory_config)
    embedding_output = ttnn.to_layout(embedding_output, layout=ttnn.TILE_LAYOUT)
    embedding_output = ttnn.add(
        embedding_output, position_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    return embedding_output


def vit_layernorm_before(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )

    return attention_output


def vit_layernorm_after(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )

    return attention_output


def vit_attention(
    config,
    hidden_states,
    attention_outputs_concatenated,
    attention_mask,
    parameters,
):
    num_heads = config.num_attention_heads
    num_heads = 12
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    # query = ttnn.linear(
    #     hidden_states,
    #     parameters.attention.query.weight,
    #     bias=parameters.attention.query.bias,
    #     memory_config=ttnn.L1_MEMORY_CONFIG, # L1_BLOCK_SHARDED_MEMORY_CONFIG
    #     dtype=ttnn.bfloat8_b,
    #     program_config=config.program_configs["query_matmul_program_config"],
    # )

    # key = ttnn.linear(
    #     hidden_states,
    #     parameters.attention.key.weight,
    #     bias=parameters.attention.key.bias,
    #     memory_config=ttnn.L1_MEMORY_CONFIG, # L1_BLOCK_SHARDED_MEMORY_CONFIG
    #     dtype=ttnn.bfloat8_b,
    #     program_config=config.program_configs["query_matmul_program_config"],
    # )

    # value = ttnn.linear(
    #     hidden_states,
    #     parameters.attention.value.weight,
    #     bias=parameters.attention.value.bias,
    #     memory_config=ttnn.L1_MEMORY_CONFIG, # L1_BLOCK_SHARDED_MEMORY_CONFIG
    #     dtype=ttnn.bfloat8_b,
    #     program_config=config.program_configs["query_matmul_program_config"],
    # )

    query_key_value = ttnn.linear(
        hidden_states,
        parameters.attention.query_key_value.weight,
        bias=parameters.attention.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,  # L1_BLOCK_SHARDED_MEMORY_CONFIG
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["query_key_value_matmul_program_config"],
    )
    # print("QKV_out", query_key_value.shape)

    (
        query,
        key,
        value,
    ) = ttnn.experimental.nlp_create_qkv_heads_vit(query_key_value, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(query_key_value)

    key_layer_transposed = ttnn.transpose(
        key,
        -2,
        -1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    key.deallocate()

    ####################
    grid_size = (8, 8)
    allowed_num_cores = 64
    num_slices = 12
    seq_len = 4096
    head_dim = 64
    slice_ratio = num_heads / num_slices

    tiles_per_shard = math.ceil((((num_heads * seq_len) / allowed_num_cores) / num_slices) / 32)
    qv_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    k_activations_height_shard_spec = [2 * 32, tiles_per_shard * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    # print("QQQ", query.shape, key_layer_transposed.shape, tiles_per_shard, qv_activations_height_shard_spec, k_activations_height_shard_spec)

    # Slice inputs and operate on each slice separately
    for i in range(num_slices):
        query_slices = ttnn.interleaved_to_sharded_partial(
            query,
            grid_size,
            qv_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        key_slices = ttnn.interleaved_to_sharded_partial(
            key_layer_transposed,
            # key,
            grid_size,
            k_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        key_slices = ttnn.to_memory_config(key_slices, memory_config=ttnn.L1_MEMORY_CONFIG)
        # slice_start = (0, int(i*slice_ratio), 0, 0)
        # slice_end   = (1, int(i*slice_ratio + slice_ratio), key.shape[-2], key.shape[-1])
        # key_slices = ttnn.slice(key, slice_start, slice_end)
        # key_slices_transposed = ttnn.transpose( key_slices, -2, -1, memory_config=ttnn.L1_MEMORY_CONFIG, )

        value_slices = ttnn.interleaved_to_sharded_partial(
            value,
            grid_size,
            qv_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        value_slices = ttnn.to_memory_config(value_slices, memory_config=ttnn.L1_MEMORY_CONFIG)

        # print("QK", query_slices.shape, key_slices.shape)
        ### QKT MATMUL ###
        mm_slices = ttnn.matmul(
            query_slices,
            key_slices,
            # key_slices_transposed,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=config.program_configs["query_by_key_matmul_program_config"],
            # compute_kernel_config=self.model_config["QKTV_AND_SOFTMAX_OPTIMIZED_KERNEL_CONFIG"],
        )
        # print(mm_slices.shape)

        mm_slices = ttnn.transformer.attention_softmax_(
            mm_slices,
            attention_mask=attention_mask,
            head_size=(num_heads // num_slices),
            program_config=config.program_configs["softmax_program_config"],
        )

        # softmax_program_config = self.model_config["SOFTMAX_OPTIMIZED_PROGCFG"](
        #     grid_size, subblock_w, mm_output_height_shard_spec[0] // 32, mm_output_height_shard_spec[1] // 32
        # )
        ### SOFTMAX ###
        # mm_slices = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
        #     mm_slices,
        #     self.scalar_for_optimized_prefill,
        #     attention_mask[i],
        #     program_config=softmax_program_config,
        #     compute_kernel_config=self.model_config["QKTV_AND_SOFTMAX_OPTIMIZED_KERNEL_CONFIG"],
        # )

        ### QKTV MATMUL ###
        # print("QK-V", mm_slices.shape, value_slices.shape)
        attn_out_slices = ttnn.matmul(
            mm_slices,
            value_slices,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=config.program_configs["attention_probabilities_by_value_matmul_program_config"],
        )

        # print("S2I", attention_outputs_concatenated.shape, attn_out_slices.shape)
        ttnn.sharded_to_interleaved_partial(
            attn_out_slices,
            attention_outputs_concatenated,
            num_slices,
            i,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attn_out_slices.deallocate(True)
        mm_slices.deallocate(True)
        query_slices.deallocate(True)
        key_slices.deallocate(True)
        value_slices.deallocate(True)

    ####################

    # print("CONC", attention_outputs_concatenated.shape)
    context_layer = ttnn.experimental.nlp_concat_heads(
        attention_outputs_concatenated,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)

    return self_output


def vit_intermediate(
    config,
    hidden_states,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff1_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    return output


def vit_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    output = ttnn.add(output, residual, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(residual)

    return output


def vit_feedforward(
    config,
    hidden_states,
    attention_output,
    *,
    parameters,
):
    # intermediate = vit_intermediate(config, hidden_states, parameters=parameters.intermediate)
    # output = vit_output(config, intermediate, attention_output, parameters=parameters.output)

    output_1 = ttnn.linear(
        hidden_states,
        parameters.intermediate.dense.weight,
        bias=parameters.intermediate.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff1_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    output = ttnn.linear(
        output_1,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(output_1)

    output = ttnn.add(output, attention_output, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(attention_output)

    return output


def vit_layer(
    config,
    hidden_states,
    attention_outputs_concatenated,
    attention_mask,
    parameters,
):
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )

    multi_head_attention_output = vit_attention(
        config,
        layernorm_before_output,
        attention_outputs_concatenated,
        attention_mask=attention_mask,
        parameters=parameters.attention,
    )

    multi_head_attention_output = ttnn.add(
        multi_head_attention_output,
        hidden_states,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_after_output_program_config"],
    )

    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        multi_head_attention_output,
        parameters=parameters,
    )

    return feedforward_output


def vit_encoder(
    config,
    embeddings,
    head_masks,
    parameters,
):
    encoder_input = ttnn.to_memory_config(
        embeddings,
        memory_config=ttnn.create_sharded_memory_config(
            [
                config.core_grid.y,
                224,
                768,
            ],  # embeddings.shape, # hardcoded because a bug where it still sees the 197 not 224
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(embeddings)

    for index, encoder_parameters in enumerate(parameters.layer):
        encoder_output = vit_layer(
            config,
            encoder_input,
            head_masks[index],
            encoder_parameters,
        )
        encoder_input = encoder_output

    return encoder_output


def vit(
    config,
    pixel_values,
    attention_mask,
    cls_token,
    position_embeddings,
    parameters,
):
    embeddings_output = vit_embeddings(config, pixel_values, cls_token, position_embeddings, parameters=parameters)

    hidden_states = vit_encoder(
        config,
        embeddings_output,
        attention_mask,
        parameters=parameters.vit.encoder,
    )

    # Final LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.vit.layernorm.weight,
        bias=parameters.vit.layernorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )

    # Classifier
    classifier_output = ttnn.linear(
        output,
        parameters.classifier.weight,
        bias=parameters.classifier.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["classifer_matmul_program_config"],
    )

    return classifier_output


def preprocess_inputs(
    input_ids,
    token_type_ids,
    attention_mask,
    device,
):
    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 0, 0, 0, 0, 0, 0, batch_size - 1))
        attention_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    return input_ids, token_type_ids, attention_mask


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.vit.modeling_vit.ViTEmbeddings):
        weight = torch_model.patch_embeddings.projection.weight
        bias = torch_model.patch_embeddings.projection.bias

        three_times_hidden_size, c, _, _ = weight.shape
        pad_value = 4 - c
        preprocessed_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
        preprocessed_weight = torch.reshape(
            preprocessed_weight, (int(three_times_hidden_size * (4 / c)), three_times_hidden_size)
        )

        parameters = {"patch_embeddings": {}}
        parameters["patch_embeddings"] = {"projection": {}}
        parameters["patch_embeddings"]["projection"]["weight"] = ttnn.from_torch(
            preprocessed_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        parameters["patch_embeddings"]["projection"]["bias"] = ttnn.from_torch(
            bias.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

        parameters["cls_token"] = ttnn.from_torch(torch_model.cls_token, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        parameters["position_embeddings"] = ttnn.from_torch(
            torch_model.position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

    if hasattr(torch_model, "query") and hasattr(torch_model, "key") and hasattr(torch_model, "value"):
        num_heads = 12
        head_size = 64
        hidden_size = num_heads * head_size * 3
        qkv_weight = torch.cat(
            [
                torch_model.query.weight.reshape([num_heads, head_size, -1]),
                torch_model.key.weight.reshape([num_heads, head_size, -1]),
                torch_model.value.weight.reshape([num_heads, head_size, -1]),
            ],
            dim=1,
        ).reshape([hidden_size, -1])
        qkv_bias = torch.cat(
            [
                torch_model.query.bias.reshape([num_heads, head_size]),
                torch_model.key.bias.reshape([num_heads, head_size]),
                torch_model.value.bias.reshape([num_heads, head_size]),
            ],
            dim=1,
        ).reshape([hidden_size])

        parameters = {"query_key_value": {}, "query": {}, "key": {}, "value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b)

        parameters["query"]["weight"] = preprocess_linear_weight(torch_model.query.weight, dtype=ttnn.bfloat8_b)
        parameters["query"]["bias"] = preprocess_linear_bias(torch_model.query.bias, dtype=ttnn.bfloat8_b)

        parameters["key"]["weight"] = preprocess_linear_weight(torch_model.key.weight, dtype=ttnn.bfloat8_b)
        parameters["key"]["bias"] = preprocess_linear_bias(torch_model.key.bias, dtype=ttnn.bfloat8_b)

        parameters["value"]["weight"] = preprocess_linear_weight(torch_model.value.weight, dtype=ttnn.bfloat8_b)
        parameters["value"]["bias"] = preprocess_linear_bias(torch_model.value.bias, dtype=ttnn.bfloat8_b)

    elif isinstance(torch_model, torch.nn.Linear):
        # TODO: better way of detection for the classify linear weights
        if torch_model.weight.shape[0] == 1000:
            preprocessed_weight = torch.nn.functional.pad(torch_model.weight, (0, 0, 0, int(1152 - 1000)))
            preprocessed_bias = torch.nn.functional.pad(torch_model.bias, (0, int(1152 - 1000)))
            parameters["weight"] = preprocess_linear_weight(preprocessed_weight, dtype=ttnn.bfloat8_b)
            parameters["bias"] = preprocess_linear_bias(preprocessed_bias, dtype=ttnn.bfloat8_b)
        else:
            parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat8_b)
            parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat8_b)

    return parameters
