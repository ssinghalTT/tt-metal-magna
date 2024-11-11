# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import pytest
import torch
from loguru import logger
import ttnn
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)
from models.demos.wormhole.bloom.tt import ttnn_optimized_bloom
from transformers import BloomForQuestionAnswering, BloomConfig, BloomTokenizerFast, BloomForCausalLM, pipeline

from models.demos.wormhole.bloom.demo.demo_utils import (
    squadv2_1K_samples_input,
    squadv2_answer_decode_batch,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import is_wormhole_b0, skip_for_grayskull
import evaluate


def load_inputs(input_path, batch):
    with open(input_path) as f:
        input_data = json.load(f)
        assert len(input_data) >= batch, f"Input data needs to have at least {batch} (batch size) entries."
        context = []
        question = []
        for i in range(batch):
            context.append(input_data[i]["context"])
            question.append(input_data[i]["question"])
        return context, question


def run_bloom_question_answering_inference(
    model_name,
    batch_size,
    sequence_size,
    bloom,
    model_location_generator,
    input_path,
    mesh_device,
):
    disable_persistent_kernel_cache()

    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    torch_model = BloomForQuestionAnswering.from_pretrained(model_name).eval()

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    profiler.start(f"preprocessing_parameter")

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=f"ttnn_functional_bloom_for_question_answering",
            initialize_model=lambda: torch_model,
            device=mesh_device,
            custom_preprocessor=ttnn_optimized_bloom.custom_preprocessor,
        )
    profiler.end(f"preprocessing_parameter")
    num_heads = config.n_head

    nlp = pipeline("question-answering", model=torch_model, tokenizer=tokenizer)
    context, question = load_inputs(input_path, batch_size)
    preprocess_params, _, postprocess_params = nlp._sanitize_parameters(max_seq_len=sequence_size, padding="max_length")
    inputs = nlp._args_parser({"question": question, "context": context})
    preprocessed_inputs = []

    for i in range(batch_size):
        model_input = next(nlp.preprocess(inputs[0][i], **preprocess_params))
        single_input = {
            "example": model_input["example"],
            "inputs": model_input,
        }
        preprocessed_inputs.append(single_input)

    bloom_input = tokenizer(
        question,
        context,
        max_length=sequence_size,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    profiler.start(f"preprocessing_input")

    input_ids, alibi, attention_mask = bloom.preprocess_inputs(
        input_ids=bloom_input["input_ids"],
        device=mesh_device,
        num_heads=num_heads,
        attention_mask=bloom_input["attention_mask"],
        max_length=sequence_size,
        mesh_mapper=inputs_mesh_mapper,
    )
    profiler.end(f"preprocessing_input")

    tt_output = bloom.bloom_for_question_answering(
        config,
        input_ids=input_ids,
        alibi=alibi,
        causal_mask=attention_mask,
        parameters=parameters,
    )
    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer).to(torch.float32)

    tt_start_logits = tt_output[..., 0]
    tt_end_logits = tt_output[..., 1]
    model_answers = {}

    profiler.start("post_processing_output_to_string")
    for i in range(batch_size):
        tt_res = {
            "start": tt_start_logits[i],
            "end": tt_end_logits[i],
            "example": preprocessed_inputs[i]["example"],
            **preprocessed_inputs[i]["inputs"],
        }
        tt_answer = nlp.postprocess([tt_res], **postprocess_params)
        logger.info(f"answer: {tt_answer['answer']}\n")
        model_answers[i] = tt_answer["answer"]
    profiler.end("post_processing_output_to_string")

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "preprocessing_input": profiler.get("preprocessing_input"),
        "inference_time": profiler.get("inference_time"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }
    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"preprocessing_input: {measurements['preprocessing_input']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")
    logger.info(f"post_processing : {measurements['post_processing']} s")
    return measurements


def run_bloom_question_answering_inference_squad_v2(
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    bloom,
    model_location_generator,
    n_iterations,
    mesh_device,
):
    disable_persistent_kernel_cache()

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

    nlp = pipeline("question-answering", model=torch_model, tokenizer=tokenizer)
    attention_mask = True
    token_type_ids = False
    inputs_squadv2 = squadv2_1K_samples_input(tokenizer, sequence_size, attention_mask, token_type_ids, batch_size)
    squad_metric = evaluate.load("squad_v2")

    with torch.no_grad():
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        i = 0
        for batch in inputs_squadv2:
            if i < n_iterations:
                batch_data = batch[0]
                curr_batch_size = batch_data["input_ids"].shape[0]
                ttnn_bloom_inputs = bloom.preprocess_inputs(
                    input_ids=batch_data["input_ids"],
                    device=mesh_device,
                    num_heads=num_heads,
                    attention_mask=batch_data["attention_mask"],
                    max_length=sequence_size,
                    mesh_mapper=inputs_mesh_mapper,
                )
                tt_output = ttnn_optimized_bloom.bloom_for_question_answering(
                    config,
                    input_ids=ttnn_bloom_inputs[0],
                    alibi=ttnn_bloom_inputs[1],
                    causal_mask=ttnn_bloom_inputs[2],
                    parameters=parameters,
                )
                tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer).to(torch.float32)

                cpu_output = torch_model(**batch_data)
                references = batch[1]
                question = batch[2]
                context = batch[3]

                cpu_predictions, tt_predictions = squadv2_answer_decode_batch(
                    torch_model,
                    tokenizer,
                    nlp,
                    references,
                    cpu_output,
                    tt_output,
                    curr_batch_size,
                    question,
                    context,
                )

                pred_labels.extend(tt_predictions)
                cpu_pred_labels.extend(cpu_predictions)
                true_labels.extend(references)
                del tt_output

                i += 1
        eval_score = squad_metric.compute(predictions=pred_labels, references=true_labels)
        cpu_eval_score = squad_metric.compute(predictions=cpu_pred_labels, references=true_labels)
        logger.info(f"\tTT_Eval: exact: {eval_score['exact']} --  F1: {eval_score['f1']}")
        logger.info(f"\tCPU_Eval: exact: {cpu_eval_score['exact']} -- F1:  {cpu_eval_score['f1']}")


@skip_for_grayskull()
@pytest.mark.parametrize(
    "model_name, input_loc",
    ((["bigscience/bloom-560m", "models/demos/wormhole/bloom/demo/input_qa.json"]),),
)
@pytest.mark.parametrize("bloom", [ttnn_optimized_bloom])
def test_demo(input_loc, model_name, bloom, model_location_generator, mesh_device):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_bloom_question_answering_inference(
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        bloom=bloom,
        model_location_generator=model_location_generator,
        input_path=input_loc,
        mesh_device=mesh_device,
    )


@skip_for_grayskull()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("bloom", [ttnn_optimized_bloom])
@pytest.mark.parametrize(
    "n_iterations",
    ((5),),
)
def test_demo_squadv2(
    model_name, bloom, n_iterations, model_location_generator, use_program_cache, mesh_device, reset_seeds
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_bloom_question_answering_inference_squad_v2(
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        bloom=bloom,
        model_location_generator=model_location_generator,
        n_iterations=n_iterations,
        mesh_device=mesh_device,
    )
