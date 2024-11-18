# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import pytest
import ttnn

from models.utility_functions import (
    profiler,
)

from models.demos.distilbert.tests.distilbert_tiny_test_infra import create_test_infra

from models.perf.perf_utils import prep_perf_report

from transformers import DistilBertForQuestionAnswering, AutoTokenizer

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


def dump_device_profiler(device):
    if isinstance(device, ttnn.Device):
        ttnn.DumpDeviceProfiler(device)
    else:
        for dev in device.get_device_ids():
            ttnn.DumpDeviceProfiler(device.get_device(dev))


# TODO: Create ttnn apis for this
ttnn.dump_device_profiler = dump_device_profiler

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}

# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


def run_trace_2cq_model(
    device,
    input_ids,
    attention_mask,
    position_ids,
    min_val_tensor,
    negative_val_tensor,
    test_infra,
    num_warmup_iterations,
    num_measurement_iterations,
):
    (
        tt_input_ids_host,
        tt_attention_mask_host,
        tt_position_ids_host,
        tt_min_val_tensor_host,
        tt_negative_val_tensor_host,
    ) = test_infra.setup_dram_sharded_input(device)
    tt_input_ids = tt_input_ids_host.to(device)
    tt_attention_mask = tt_attention_mask_host.to(device)
    tt_position_ids = tt_position_ids_host.to(device)
    tt_min_val_tensor = tt_min_val_tensor_host.to(device)
    tt_negative_val_tensor = tt_negative_val_tensor_host.to(device)

    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    profiler.start("compile")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)
    ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_min_val_tensor_host, tt_min_val_tensor, 1)
    ttnn.copy_host_to_device_tensor(tt_negative_val_tensor_host, tt_negative_val_tensor, 1)

    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_ids = tt_input_ids  # ttnn.to_memory_config(tt_input_ids, input_mem_config)
    test_infra.position_ids = tt_position_ids
    test_infra.attention_mask = tt_attention_mask
    test_infra.negative_val_tensor = tt_negative_val_tensor
    test_infra.min_val_tensor = tt_min_val_tensor
    shape = test_infra.input_ids.shape
    dtype = test_infra.input_ids.dtype
    layout = test_infra.input_ids.layout
    ttnn.record_event(0, op_event)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)

    profiler.start("cache")
    ttnn.wait_for_event(1, op_event)
    # ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)
    ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_min_val_tensor_host, tt_min_val_tensor, 1)
    ttnn.copy_host_to_device_tensor(tt_negative_val_tensor_host, tt_negative_val_tensor, 1)

    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    # test_infra.input_tensor = ttnn.to_memory_config(tt_input_ids, input_mem_config)
    test_infra.input_tensor = tt_input_ids  # ttnn.to_memory_config(tt_input_ids, input_mem_config)
    test_infra.position_ids = tt_position_ids
    test_infra.attention_mask = tt_attention_mask
    test_infra.negative_val_tensor = tt_negative_val_tensor
    test_infra.min_val_tensor = tt_min_val_tensor
    ttnn.record_event(0, op_event)
    # Deallocate the previous output tensor here to make allocation match capture setup
    # This allows us to allocate the input tensor after at the same address

    test_infra.output_tensor.deallocate(force=True)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    # Capture
    ttnn.wait_for_event(1, op_event)
    # ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)
    ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
    ttnn.copy_host_to_device_tensor(tt_min_val_tensor_host, tt_min_val_tensor, 1)
    ttnn.copy_host_to_device_tensor(tt_negative_val_tensor_host, tt_negative_val_tensor, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    # test_infra.input_tensor = ttnn.to_memory_config(tt_input_ids, input_mem_config)
    test_infra.input_tensor = tt_input_ids  # ttnn.to_memory_config(tt_input_ids, input_mem_config)
    test_infra.position_ids = tt_position_ids
    test_infra.attention_mask = tt_attention_mask
    test_infra.negative_val_tensor = tt_negative_val_tensor
    test_infra.min_val_tensor = tt_min_val_tensor
    ttnn.record_event(0, op_event)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = test_infra.run()
    # input_tensor = ttnn.allocate_tensor_on_device(
    #     shape,
    #     dtype,
    #     layout,
    #     device,
    #     input_mem_config,
    # )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    # assert trace_input_addr == ttnn.buffer_address(input_tensor)
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        # ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)
        ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_min_val_tensor_host, tt_min_val_tensor, 1)
        ttnn.copy_host_to_device_tensor(tt_negative_val_tensor_host, tt_negative_val_tensor, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        # input_tensor = ttnn.reshard(tt_input_ids, input_mem_config, input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        ttnn.dump_device_profiler(device)

    ttnn.synchronize_devices(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.wait_for_event(1, op_event)
        # ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_input_ids_host, tt_input_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_attention_mask_host, tt_attention_mask, 1)
        ttnn.copy_host_to_device_tensor(tt_position_ids_host, tt_position_ids, 1)
        ttnn.copy_host_to_device_tensor(tt_min_val_tensor_host, tt_min_val_tensor, 1)
        ttnn.copy_host_to_device_tensor(tt_negative_val_tensor_host, tt_negative_val_tensor, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        # TODO: Add in place support to ttnn to_memory_config
        # input_tensor = ttnn.reshard(tt_input_ids, input_mem_config, input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(tt_output_res.cpu(blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)

    ttnn.release_trace(device, tid)


def run_perf_distilbert(device_batch_size, expected_inference_time, expected_compile_time, device, modek_version):
    profiler.clear()

    if device_batch_size <= 2:
        pytest.skip("Batch size 1 and 2 are not supported with sharded data")

    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    num_devices = device.get_num_devices() if is_mesh_device else 1
    batch_size = device_batch_size * num_devices
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "distilbert-base-uncased-distilled-squad"

    HF_model = DistilBertForQuestionAnswering.from_pretrained(model_name).eval()
    config = config = HF_model.config

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    question = batch_size * ["Where do I live?"]
    context = batch_size * ["My name is Merve and I live in İstanbul."]
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        padding="max_length",
        max_length=384,
        truncation=True,
        return_attention_mask=True,
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
    mask_reshp = (batch_size, 1, 1, attention_mask.shape[1])
    score_shape = (batch_size, 12, 384, 384)

    mask = (attention_mask == 0).view(mask_reshp).expand(score_shape)
    min_val = torch.zeros(score_shape)
    min_val_tensor = min_val.masked_fill(mask, torch.tensor(torch.finfo(torch.bfloat16).min))

    negative_val = torch.zeros(score_shape)
    negative_val_tensor = negative_val.masked_fill(mask, -1)

    test_infra = create_test_infra(
        device,
        device_batch_size,
        model_config["ACTIVATIONS_DTYPE"],
        model_config["WEIGHTS_DTYPE"],
        model_config["MATH_FIDELITY"],
        config=config,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.synchronize_devices(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = HF_model(input_ids, attention_mask)
        profiler.end(cpu_key)

        run_trace_2cq_model(
            device,
            input_ids,
            attention_mask,
            position_ids,
            min_val_tensor,
            negative_val_tensor,
            test_infra,
            num_warmup_iterations,
            num_measurement_iterations,
        )
    first_iter_time = profiler.get(f"compile") + profiler.get(f"cache")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_measurement_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - 2 * inference_time_avg
    comments = "distilbert-base-uncased-distilled-squad"
    prep_perf_report(
        model_name=f"ttnn_distilbert{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(
        f"{model_name} {comments} inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"{model_name} compile time: {compile_time}")
