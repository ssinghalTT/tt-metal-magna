import ttnn
import numpy as np
import torch

from transformers import AutoModelForCausalLM
from models.demos.t3000.llama2_70b.reference.llama.llama.model import precompute_freqs_cis, apply_rotary_emb
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rope
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from models.tt_transformers.tt.rope import RotarySetup

q_before = np.load("/home/ubuntu/intermediate_results/query_states_before_rope.npy")
q_after = np.load("/home/ubuntu/intermediate_results/query_states_after_rope.npy")

model_name = "TinyLlama/TinyLlama_v1.1"
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
hf_q_proj = hf_model.state_dict()["model.layers.0.self_attn.q_proj.weight"]

import torch
import math


def do_meta_rope(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Applies Meta-style RoPE to an input tensor using precomputed frequencies.
    Assumes input tensor 'x' has an interleaved layout for the head dimension.

    Args:
        x: Input tensor, shape e.g., [batch, seq_len, num_heads, head_dim]
           The last dimension (head_dim) must be interleaved.
        freqs_cis: Precomputed complex frequency tensor (cos + i*sin),
                   shape [seq_len, 1, head_dim // 2], or broadcastable.

    Returns:
        torch.Tensor: Rotated tensor with the same shape as x.
    """
    # Get shape details
    batch_size, seq_len, num_heads, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    rotated, _ = apply_rotary_emb(x, x, freqs_cis)
    return rotated


def interleave_halves(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Convert contiguous split‑half layout  ➜  interleaved real/imag layout
    by hopping through NumPy (always contiguous).

    [a0 … an‑1 | b0 … bn‑1] → [a0, b0, a1, b1, …]
    """
    # Normalise dim to positive
    dim = dim if dim >= 0 else x.dim() + dim
    d = x.size(dim)
    assert d % 2 == 0, "hidden dim must be even"

    # --- numpy hop ---
    np_arr = x.detach().cpu().numpy()  # (forces contiguous)
    a, b = np.split(np_arr, 2, axis=dim)
    inter = np.stack((a, b), axis=dim + 1).reshape(np_arr.shape)

    # back to torch, on original device & dtype
    return torch.tensor(inter, dtype=x.dtype, device=x.device)


def deinterleave_halves(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Inverse of interleave_halves, again via NumPy.
    Interleaved real/imag ➜ contiguous split‑half.
    """
    dim = dim if dim >= 0 else x.dim() + dim
    d = x.size(dim)
    assert d % 2 == 0

    np_arr = x.detach().cpu().numpy()
    # shape: (..., d) -> (..., d//2, 2)
    resh = np_arr.reshape(*np_arr.shape[:dim], d // 2, 2, *np_arr.shape[dim + 1 :])
    a, b = np.split(resh, 2, axis=dim + 1)  # split the 2‑axis
    deint = np.concatenate((a.squeeze(dim + 1), b.squeeze(dim + 1)), axis=dim)

    return torch.tensor(deint, dtype=x.dtype, device=x.device)


def do_hf_rope(queries):
    assert len(queries.shape) == 4
    cache_position = None
    past_seen_tokens = 0
    if cache_position is None:
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + queries.shape[1], device=queries.device)
    position_ids = None
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    print(position_ids.shape)
    head_dim = queries.shape[-1]
    seq_len = queries.shape[2]
    rope_calc = LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=seq_len, base=10000)
    cos, sin = rope_calc(queries, position_ids)
    roped_queries, _ = hf_apply_rope(queries, queries, cos, sin, unsqueeze_dim=2)
    return roped_queries


def unpermute_proj(w, n_heads=32):
    """
    Undo Hugging Face's permute() on q/k projection weights so they work
    with Meta‑style interleaved RoPE (or Tenstorrent's kernel).

    w: [dim, dim] or [out_dim, in_dim] weight matrix
    n_heads: number of heads that produced this weight
    (use n_kv_heads for k_proj in GQA models)
    """
    dim = w.shape[0]  # Llama weights are square
    # view -> (n_heads, 2, dim/(n_heads*2), dim)
    # swap the 2‑axis back to position 2, then reshape flat again
    return w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)  # (n_heads, dim/… , 2, dim)


print("opening device")
device = ttnn.open_device(device_id=0)
print("opened device")


def do_tt_rope(meta_q, scale_factor=None):
    meta_q = meta_q.transpose(1, 2)
    assert len(meta_q.shape) == 4
    head_dim = meta_q.shape[-1]
    seq_len = meta_q.shape[2]
    rope_setup = RotarySetup(
        device,
        batch_size=1,
        head_dim=head_dim,
        max_seq_len=seq_len,
        rope_theta=10000,
        scale_factor=scale_factor,
        orig_context_len=2048,
        datatype=ttnn.bfloat16,
    )

    # Slice the rot mats to the prefill seqlen
    cos_freqs, sin_freqs = [rope_setup.cos_matrix[:, :, :seq_len, :], rope_setup.sin_matrix[:, :, :seq_len, :]]

    cos_freqs = ttnn.to_layout(cos_freqs, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    sin_freqs = ttnn.to_layout(sin_freqs, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    from models.tt_transformers.tt.common import get_rot_transformation_mat

    trans_mat = get_rot_transformation_mat(dhead=32)

    trans_mat = ttnn.from_torch(trans_mat, device=device)

    trans_mat = ttnn.to_layout(trans_mat, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    queries = ttnn.from_torch(meta_q, device=device, dtype=ttnn.bfloat16)
    queries = ttnn.to_layout(queries, ttnn.TILE_LAYOUT)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        # math_fidelity=ttnn.MathFidelity.LoFi,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=(True if head_dim <= 128 else False),
        packer_l1_acc=True,
    )

    ret = ttnn.experimental.rotary_embedding_llama(
        queries, cos_freqs, sin_freqs, trans_mat, is_decode_mode=False, compute_kernel_config=compute_kernel_config
    )
    ret = ttnn.to_torch(ret)
    ret = ret.transpose(1, 2)
    return ret


def compare_ropes(seq_len, scale_factor=None, theta=10000.0, num_heads=32, head_dim=64):
    print(f"Running comparison with head_dim={head_dim}")
    hf_q = torch.randn(1, seq_len, num_heads, head_dim)
    hf_roped = do_hf_rope(hf_q)
    hf_roped_interleaved = interleave_halves(hf_roped)

    # assert hf_model.config.hidden_size // hf_model.config.num_attention_heads == head_dim
    # assert hf_model.config.rope_theta == 10000.0
    # assert hf_model.config.rope_scaling == None

    def check_proj_permute_path():
        num_heads = 32
        head_dim = 64
        x_hidden = torch.randn(1, seq_len, head_dim * num_heads)
        hf_q = torch.einsum("cd,bsc->bsd", hf_q_proj, x_hidden)
        hf_q = hf_q.reshape(1, seq_len, num_heads, head_dim)
        hf_roped = do_hf_rope(hf_q)
        hf_roped_interleaved = interleave_halves(hf_roped)
        meta_q_proj = unpermute_proj(hf_q_proj, n_heads=32)
        meta_q = torch.einsum("cd,bsc->bsd", meta_q_proj, x_hidden)
        meta_q = meta_q.reshape(1, seq_len, num_heads, head_dim)
        q_after_actual = do_tt_rope(meta_q)
        q_after_actual = q_after_actual.float().cpu().numpy()
        hf_ref = hf_roped_interleaved.float().cpu().numpy()
        mean_abs_error = np.mean(np.abs(hf_ref - q_after_actual))
        print(f"proj permute: Mean absolute error: {mean_abs_error}")
        print(f"proj permute: Max absolute error: {np.max(np.abs(hf_ref - q_after_actual))}")

    def check_interleave_path():
        interleaved_hf_query = interleave_halves(hf_q)
        q_after_actual_with_interleave = do_tt_rope(interleaved_hf_query, scale_factor=scale_factor)
        q_after_actual_with_interleave = q_after_actual_with_interleave.float().cpu().numpy()
        hf_ref = hf_roped_interleaved.float().cpu().numpy()
        mean_abs_error = np.mean(np.abs(hf_ref - q_after_actual_with_interleave))
        print(f"interleave: Mean absolute error: {mean_abs_error}")
        print(f"interleave: Max absolute error: {np.max(np.abs(hf_ref - q_after_actual_with_interleave))}")

    def check_meta_path():
        interleaved_hf_query = interleave_halves(hf_q)
        q_after_actual_with_interleave = do_meta_rope(interleaved_hf_query)
        q_after_actual_with_interleave = q_after_actual_with_interleave.float().cpu().numpy()
        hf_ref = hf_roped_interleaved.float().cpu().numpy()
        mean_abs_error = np.mean(np.abs(hf_ref - q_after_actual_with_interleave))
        print(f"meta: Mean absolute error: {mean_abs_error}")
        print(f"meta: Max absolute error: {np.max(np.abs(hf_ref - q_after_actual_with_interleave))}")

    check_proj_permute_path()
    check_interleave_path()
    check_meta_path()


compare_ropes(32, head_dim=32, scale_factor=None)


def check_meta_tt_bf16():
    x = torch.randn(1, 32, 32, 64, dtype=torch.bfloat16)
    meta_roped = do_meta_rope(x.to(torch.float32))
    tt_roped = do_tt_rope(x, scale_factor=None)
    mean_abs_error = (meta_roped - tt_roped).abs().mean().item()
    print(f"Mean absolute error: {mean_abs_error}")


check_meta_tt_bf16()
