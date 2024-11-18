# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import pytest
import torch
import torchvision
import transformers
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.demos.distilbert.tt.ttnn_optimized_distilbert import distilbert_for_question_answering
from models.demos.distilbert.tt import ttnn_optimized_distilbert
from transformers import DistilBertForQuestionAnswering, AutoTokenizer


class DistilbertTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        dealloc_input,
        final_output_mem_config,
        model_version,
        config,
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.math_fidelity = math_fidelity
        self.dealloc_input = dealloc_input
        self.final_output_mem_config = final_output_mem_config
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.config = config
        HF_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        model = HF_model.eval()
        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_optimized_distilbert.custom_preprocessor,
            convert_to_ttnn=lambda *_: True,
        )
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
        self.input_ids = inputs.input_ids
        self.attention_mask = inputs.attention_mask
        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        mask_reshp = (batch_size, 1, 1, self.attention_mask.shape[1])
        score_shape = (batch_size, 12, 384, 384)

        self.mask = (self.attention_mask == 0).view(mask_reshp).expand(score_shape)
        self.min_val = torch.zeros(score_shape)
        self.min_val_tensor = self.min_val.masked_fill(self.mask, torch.tensor(torch.finfo(torch.bfloat16).min))

        self.negative_val = torch.zeros(score_shape)
        self.negative_val_tensor = self.negative_val.masked_fill(self.mask, -1)
        torch_output = model(self.input_ids, self.attention_mask)

        model_config = {
            "MATH_FIDELITY": math_fidelity,
            "WEIGHTS_DTYPE": weight_dtype,
            "ACTIVATIONS_DTYPE": act_dtype,
        }

        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        self.distilbert_model = distilbert_for_question_answering

        self.ops_parallel_config = {}

    def get_mesh_mappers(self, device):
        is_mesh_device = isinstance(device, ttnn.MeshDevice)
        if is_mesh_device:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = ttnn.ReplicateTensorToMesh(
                device
            )  # causes unnecessary replication/takes more time on the first pass
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def setup_l1_sharded_input(
        self, device, input_ids, attention_mask, position_ids, min_val_tensor, negative_val_tensor
    ):
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()

        tt_input_ids_host = ttnn.from_torch(
            input_ids, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )

        tt_attention_mask_host = ttnn.from_torch(
            attention_mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )

        tt_position_ids_host = ttnn.from_torch(
            position_ids, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )

        tt_min_val_tensor_host = ttnn.from_torch(
            min_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )

        tt_negative_val_tensor_host = ttnn.from_torch(
            negative_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )

        return (
            tt_input_ids_host,
            tt_attention_mask_host,
            tt_position_ids_host,
            tt_min_val_tensor_host,
            tt_negative_val_tensor_host,
        )

    def setup_dram_sharded_input(self, device):
        (
            tt_input_ids_host,
            tt_attention_mask_host,
            tt_position_ids_host,
            tt_min_val_tensor_host,
            tt_negative_val_tensor_host,
        ) = self.setup_l1_sharded_input(
            device,
            self.input_ids,
            self.attention_mask,
            self.position_ids,
            self.min_val_tensor,
            self.negative_val_tensor,
        )

        return (
            tt_input_ids_host,
            tt_attention_mask_host,
            tt_position_ids_host,
            tt_min_val_tensor_host,
            tt_negative_val_tensor_host,
        )

    def run(self, tt_input_tensor=None):
        self.output_tensor = self.distilbert_model(
            self.config,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            parameters=self.parameters,
            device=self.device,
            min_val_tensor=self.min_val_tensor,
            negative_val_tensor=self.negative_val_tensor,
        )
        return self.output_tensor


def create_test_infra(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    config,
    use_pretrained_weight=True,
    dealloc_input=True,
    final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    model_location_generator=None,
):
    return DistilbertTestInfra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        dealloc_input,
        final_output_mem_config,
        "distilbert-base-uncased-distilled-squad",
        config,
    )
