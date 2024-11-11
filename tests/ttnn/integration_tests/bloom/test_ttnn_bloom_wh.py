# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import BloomForQuestionAnswering, BloomConfig, BloomTokenizerFast, BloomForCausalLM
from transformers import AutoTokenizer
from models.demos.wormhole.bloom.tt import ttnn_optimized_bloom
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0, skip_for_grayskull


def torch_random(shape, low, high, dtype):
    if dtype in {torch.bool, torch.int64}:
        return torch.randint(low, high, shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype).uniform_(low, high)


# real inputs
@skip_for_grayskull()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_for_question_answering_real(mesh_device, model_name, batch_size, sequence_size, reset_seeds):
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    torch_model = BloomForQuestionAnswering.from_pretrained(model_name).eval()

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=f"ttnn_functional_bloom_for_question_answering",
            initialize_model=lambda: torch_model,
            device=mesh_device,
            custom_preprocessor=ttnn_optimized_bloom.custom_preprocessor,
        )

    num_heads = config.n_head

    question = "Chopin's performances were known for what?"
    context = "All of Chopin's compositions include the piano. Most are for solo piano, though he also wrote two piano concertos, a few chamber pieces, and some songs to Polish lyrics. His keyboard style is highly individual and often technically demanding; his own performances were noted for their nuance and sensitivity. Chopin invented the concept of instrumental ballade. His major piano works also include mazurkas, waltzes, nocturnes, polonaises, études, impromptus, scherzos, preludes and sonatas, some published only after his death. Influences on his compositional style include Polish folk music, the classical tradition of J. S. Bach, Mozart and Schubert, the music of all of whom he admired, as well as the Paris salons where he was a frequent guest. His innovations in style, musical form, and harmony, and his association of music with nationalism, were influential throughout and after the late Romantic period."
    inputs = tokenizer(question, context, return_tensors="pt")

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    num_tokens = input_ids.shape[-1]
    input_ids = input_ids.expand((batch_size, num_tokens))
    attention_mask = attention_mask.expand((batch_size, num_tokens))

    torch_output = torch_model(input_ids=input_ids, attention_mask=attention_mask)
    torch_start_logits = torch_output.start_logits
    torch_end_logits = torch_output.end_logits

    input_ids, alibi, causal_mask = ttnn_optimized_bloom.preprocess_inputs(
        input_ids=input_ids,
        device=mesh_device,
        num_heads=num_heads,
        attention_mask=attention_mask,
        max_length=sequence_size,
        mesh_mapper=inputs_mesh_mapper,
    )

    tt_output = ttnn_optimized_bloom.bloom_for_question_answering(
        config, input_ids, alibi, causal_mask, parameters=parameters
    )

    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)

    tt_start_logits = tt_output[:, :num_tokens, 0]
    tt_end_logits = tt_output[:, :num_tokens, 1]

    assert_with_pcc(torch_start_logits, tt_start_logits, 0.9488497368237151)
    assert_with_pcc(torch_end_logits, tt_end_logits, 0.9271437577898639)


# real inputs
@skip_for_grayskull()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_for_causal_lm_real(mesh_device, model_name, batch_size, sequence_size, reset_seeds):
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    torch_model = BloomForCausalLM.from_pretrained(model_name).eval()

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=f"ttnn_functional_bloom_for_question_answering",
            initialize_model=lambda: torch_model,
            device=mesh_device,
            custom_preprocessor=ttnn_optimized_bloom.custom_preprocessor,
        )

    num_heads = config.n_head

    input_text = (
        "Artificial intelligence is transforming the world, improving everything from healthcare to transportation."
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    num_tokens = inputs.shape[-1]
    inputs = inputs.expand((batch_size, num_tokens))
    torch_output = torch_model(input_ids=inputs, attention_mask=None)

    input_ids, alibi, causal_mask = ttnn_optimized_bloom.preprocess_inputs(
        input_ids=inputs,
        device=mesh_device,
        num_heads=num_heads,
        attention_mask=None,
        max_length=sequence_size,
        mesh_mapper=inputs_mesh_mapper,
    )

    tt_output = ttnn_optimized_bloom.bloom_for_causal_lm(config, input_ids, alibi, causal_mask, parameters=parameters)

    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)

    tt_start_logits = tt_output[
        :batch_size,
        :num_tokens,
        :,
    ]

    assert_with_pcc(torch_output.logits, tt_start_logits, 0.999)


# random inputs
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [320])  # test is being killed for seq size > the specified seq_size
def test_bloom_for_question_answering(mesh_device, model_name, batch_size, sequence_size, reset_seeds):
    config = BloomConfig.from_pretrained(model_name)
    config.position_embedding_type = "none"

    model = BloomForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_output = model(input_ids=torch_input_ids, attention_mask=torch_attention_mask)

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=mesh_device,
            custom_preprocessor=ttnn_optimized_bloom.custom_preprocessor,
        )
    padded_input_ids, alibi, causal_mask = ttnn_optimized_bloom.preprocess_inputs(
        input_ids=torch_input_ids,
        attention_mask=torch_attention_mask,
        num_heads=config.n_head,
        device=mesh_device,
        max_length=sequence_size,
        mesh_mapper=inputs_mesh_mapper,
    )

    output = ttnn_optimized_bloom.bloom_for_question_answering(
        config,
        padded_input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)

    start_logits, end_logits = output.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()

    assert_with_pcc(torch_output.start_logits, start_logits, 0.9466580152486184)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.9500190891530871)


# random inputs
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [64])  # test is being killed for seq size > the specified seq_size
def test_bloom_for_causal_lm(mesh_device, model_name, batch_size, sequence_size, reset_seeds):
    config = BloomConfig.from_pretrained(model_name)
    model = BloomForCausalLM.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_output = model(input_ids=torch_input_ids, attention_mask=None)

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=f"ttnn_functional_bloom_for_question_answering",
            initialize_model=lambda: model,
            device=mesh_device,
            custom_preprocessor=ttnn_optimized_bloom.custom_preprocessor,
        )

    padded_input_ids, alibi, causal_mask = ttnn_optimized_bloom.preprocess_inputs(
        input_ids=torch_input_ids,
        device=mesh_device,
        num_heads=config.n_head,
        attention_mask=None,
        max_length=sequence_size,
        mesh_mapper=inputs_mesh_mapper,
    )

    output = ttnn_optimized_bloom.bloom_for_causal_lm(
        config,
        padded_input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
    assert_with_pcc(torch_output.logits, output, 0.9426283733103926)
