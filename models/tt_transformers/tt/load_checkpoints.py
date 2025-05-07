# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import re
import os
import torch
from safetensors.torch import load_file as safetensors_load_file
from tqdm import tqdm
import json
from pathlib import Path
from loguru import logger


# TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
def load_hf_state_dict(ckpt_dir):
    # First check if index file exists
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Multi-file case: Read the index file and load all referenced safetensor files
        with open(index_path, "r") as f:
            index_data = json.load(f)

        # Retrieve the weight file names from the index JSON
        weight_map = index_data["weight_map"]
        safetensor_files = set(weight_map.values())

        # Read each safetensors file mentioned in the index
        loaded_weights = {}
        for file in safetensor_files:
            safetensor_path = os.path.join(ckpt_dir, file)
            weights = safetensors_load_file(safetensor_path)
            loaded_weights.update(weights)  # Merge weights into a single dictionary
    else:
        # Single-file case: Load the single model.safetensors file
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        if not os.path.exists(safetensor_path):
            raise FileNotFoundError(f"Neither model.safetensors.index.json nor model.safetensors found in {ckpt_dir}")
        loaded_weights = safetensors_load_file(safetensor_path)

    return loaded_weights


def standardize_hf_keys(state_dict):
    if "model.embed_tokens.weight" in state_dict and "lm_head.weight" not in state_dict:
        # Assume tied to the embeddings if not present
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    # Standardize keys used in vision parts of Qwen2.5-VL
    replace_whole_name = lambda pattern, repl: lambda s: re.sub(rf"(^|\.)({pattern})($|\.)", rf"\1{repl}\3", s)
    output = {}
    for k, v in state_dict.items():
        k = replace_whole_name("qkv", "qkv_proj")(k)
        k = replace_whole_name("proj", "o_proj")(k)
        k = replace_whole_name("attn", "self_attn")(k)
        output[k] = v
    return output


def convert_hf_to_meta(state_dict, head_dim):
    state_dict = split_hf_keys(state_dict)
    state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
    state_dict = map_hf_to_meta_keys(state_dict)
    return state_dict


def load_meta_state_dict(ckpt_dir, n_layers=None, start_layer_idx=0):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    is_chunked = "layers_" in str(checkpoints[0])
    if is_chunked:
        checkpoint = load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx)
    else:
        checkpoint = load_sharded_checkpoints(checkpoints, n_layers)

    return checkpoint


def load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx):
    checkpoint = {}

    (f"Loading {len(checkpoints)} chunked checkpoint files")
    for ckpt in tqdm(checkpoints):
        if n_layers:
            # Layer range is in the file name, like layers_start-end.pth
            layer_range = ckpt.stem.split("_")[1]
            start_layer, end_layer = map(int, layer_range.split("-"))
            if start_layer > n_layers + start_layer_idx:
                continue
            if end_layer < start_layer_idx:
                continue

        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        checkpoint.update(loaded_ckpt)
    return checkpoint


def is_param_replicated_across_shards(key: str) -> bool:
    """
    Return `True` if the parameter is replicated (i.e., not sharded)
    across checkpoint files and should not be concatenated.
    """
    if key.startswith("vision_model."):
        return any(keyword in key for keyword in ("ln", "gate", "embed", "c_proj.bias"))
    else:
        # for Meta checkpoint keys, key either starts with "text_model." or contains no such prefix; both cases are handled here
        return any(keyword in key for keyword in ("norm", "gate"))


def load_sharded_checkpoints(checkpoints, n_layers):
    checkpoint = {}
    logger.info(f"Loading {len(checkpoints)} sharded checkpoint files")
    for ckpt in tqdm(checkpoints):
        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        for key, value in loaded_ckpt.items():
            if "layers." in key:
                layer_num = int(key.split("layers.")[1].split(".")[0])
                if n_layers and layer_num >= n_layers:
                    continue
            if key in checkpoint:
                checkpoint[key] += [value]
            else:
                checkpoint[key] = [value]
        del loaded_ckpt

    # concat checkpoint values
    for key, value in checkpoint.items():
        if len(value) == 1 or is_param_replicated_across_shards(key):
            checkpoint[key] = value[0]
        else:
            if key.endswith("tok_embeddings.weight") or key.endswith("output.weight"):
                assert value[0].shape[1] == 8192  # FIXME: do we need this hardcoded shape?
                # Concatenate along dimension 0 for llama3 token embeddings weight and lm head
                checkpoint[key] = torch.cat(value, dim=0)
            else:
                # cat_dim is index of the smallest dimension in value[0].shape
                cat_dim = torch.argmin(torch.tensor(value[0].shape))
                checkpoint[key] = torch.cat(value, dim=cat_dim)

    return checkpoint


def split_hf_keys(loaded_weights):
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "qkv_proj" in key:
            # split Q, K and V
            q_key = key.replace("qkv_proj", "q_proj")
            k_key = key.replace("qkv_proj", "k_proj")
            v_key = key.replace("qkv_proj", "v_proj")
            q_tensor, k_tensor, v_tensor = torch.split(tensor, tensor.shape[0] // 3, dim=0)
            converted_weights[q_key] = q_tensor
            converted_weights[k_key] = k_tensor
            converted_weights[v_key] = v_tensor
        elif "gate_up_proj" in key:
            # Split Gate and Up
            gate_key = key.replace("gate_up_proj", "gate_proj")
            up_key = key.replace("gate_up_proj", "up_proj")
            gate_tensor, up_tensor = torch.split(tensor, tensor.shape[0] // 2, dim=0)
            converted_weights[gate_key] = gate_tensor
            converted_weights[up_key] = up_tensor
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor
    return converted_weights


def convert_hf_qkv_to_meta_format(loaded_weights, head_dim):
    """Convert HuggingFace QKV weights to Meta format for RoPE compatibility."""
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "q_proj.weight" in key or "k_proj.weight" in key:
            # For weights: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = reverse_permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
        elif "q_proj.bias" in key or "k_proj.bias" in key:
            # For biases: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = reverse_permute(tensor, n_heads, tensor.shape[0], 1).squeeze(-1)
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor
    return converted_weights


def convert_meta_to_hf(state_dict, head_dim):
    state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_meta_to_hf_keys(state_dict)
    return state_dict


def replace_keys(state_dict, replacements):
    """
    Replacements are in the form (pattern, replacement).
    Patterns can use ^ to match the start of the string but are otherwise
    matched as whole words. These are not regular expressions, e.g. . is not
    a special character.
    """
    for pattern, replacement in replacements:
        pre = r"^" if pattern.startswith("^") else r"(?=^|\b)"
        post = r"\." if pattern.endswith(".") else r"(?=\b|$)"
        pattern = pattern[1:] if pattern.startswith("^") else pattern
        pattern = pattern[:-1] if pattern.endswith(".") else pattern
        pattern = pre + pattern + post
        state_dict = {re.sub(pattern, replacement, k): v for k, v in state_dict.items()}
    return state_dict


def map_hf_to_meta_keys(loaded_weights):
    """
    Map Hugging Face checkpoint keys to Meta checkpoint keys.
    You can use this to support other models by adding more mappings.
    See replace_keys for more details on the format of replacements.
    """
    replacements = [
        ("^emb.weight", "weight"),
        ("model.", ""),
        ("embed_tokens", "tok_embeddings"),
        ("lm_head", "output"),
        ("input_layernorm", "attention_norm"),
        ("post_attention_layernorm", "ffn_norm"),
        ("self_attn", "attention"),
        ("mlp", "feed_forward"),
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
        ("q_proj", "wq"),
        ("k_proj", "wk"),
        ("v_proj", "wv"),
        ("o_proj", "wo"),
    ]
    return replace_keys(loaded_weights, replacements)


def map_meta_to_hf_keys(loaded_weights):
    """
    Map Meta checkpoint keys to Hugging Face checkpoint keys FOR UNIT TESTS.
    You can use this to support other models by adding more mappings.
    See replace_keys for more details on the format of replacements.
    """
    replacements = [
        ("^tok_embeddings", "model.embed_tokens"),
        ("^norm", "model.norm"),
        # ("^weight", "emb.weight"), don't include this or no module tests can load "weights"
        ("^emb.weight", "weight"),  # unit test for embedding module does not include "emb"
        ("^layers", "model.layers"),
        ("output", "lm_head"),
        ("attention_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("attention", "self_attn"),
        ("feed_forward", "mlp"),
        ("w1", "gate_proj"),
        ("w2", "down_proj"),
        ("w3", "up_proj"),
        ("wq", "q_proj"),
        ("wk", "k_proj"),
        ("wv", "v_proj"),
        ("wo", "o_proj"),
    ]
    return replace_keys(loaded_weights, replacements)


def convert_meta_qkv_to_hf_format(loaded_weights, head_dim):
    """Convert Meta QKV weights back to HuggingFace format."""
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "wq.weight" in key or "wk.weight" in key:
            # For weights: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
        elif "wq.bias" in key or "wk.bias" in key:
            # For biases: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = permute(tensor.unsqueeze(-1), n_heads, tensor.shape[0], 1).squeeze(-1)
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor
    return converted_weights


def reverse_permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert_rope_style_hf_to_meta(cos_hf: torch.Tensor, sin_hf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts RoPE cos/sin tensors from Hugging Face style (half-dim duplicated)
    to Meta style (pairwise duplicated / odd-even interleaved).

    Args:
        cos_hf: Cosine tensor in HF format [..., seq_len, head_dim]
                (e.g., [c0, c1, ..., c_{d/2-1}, c0, c1, ..., c_{d/2-1}])
        sin_hf: Sine tensor in HF format [..., seq_len, head_dim]
                (e.g., [s0, s1, ..., s_{d/2-1}, s0, s1, ..., s_{d/2-1}])

    Returns:
        A tuple containing (cos_meta, sin_meta) in Meta format [..., seq_len, head_dim]
        (e.g., [c0, c0, c1, c1, ..., c_{d/2-1}, c_{d/2-1}],
         [s0, s0, s1, s1, ..., s_{d/2-1}, s_{d/2-1}])
    """
    # Input validation (optional but good practice)
    if cos_hf.shape != sin_hf.shape:
        raise ValueError("cos_hf and sin_hf must have the same shape.")
    if len(cos_hf.shape) < 2:
        raise ValueError("Input tensors must have at least 2 dimensions (seq_len, head_dim).")

    head_dim = cos_hf.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"Head dimension ({head_dim}) must be even.")

    half_head_dim = head_dim // 2

    # Select the first half (contains the unique frequencies)
    cos_unique = cos_hf[..., :half_head_dim]
    sin_unique = sin_hf[..., :half_head_dim]

    # Repeat each unique frequency pairwise
    cos_meta = torch.repeat_interleave(cos_unique, repeats=2, dim=-1)
    sin_meta = torch.repeat_interleave(sin_unique, repeats=2, dim=-1)

    return cos_meta, sin_meta
