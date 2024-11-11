# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.utility_functions import comp_allclose
from loguru import logger

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    to_torch,
    to_ttnn,
)
from models.utility_functions import skip_for_grayskull, skip_for_blackhole


def tt_layer_norm(
    input,
    *,
    normalized_dims=1,
    eps=1e-5,
    gamma=None,
    beta=None,
    dtype=ttnn.bfloat16,
    device=None,
    compute_kernel_config=None,
    create_mean_rstd=True,
):
    input_shape = list(input.shape)

    # mean_rstd_shape
    mean_rstd_shape = input_shape[:-normalized_dims]

    # dtype
    cpu_dtype = torch.bfloat16
    npu_dtype = dtype

    # input
    npu_input = to_ttnn(input, device=device, dtype=npu_dtype)

    # output
    output = torch.empty_like(input)
    npu_output = to_ttnn(output, device=device, dtype=npu_dtype)

    # gamma
    npu_gamma = to_ttnn(gamma, device=device, dtype=npu_dtype)

    # beta
    npu_beta = to_ttnn(beta, device=device, dtype=npu_dtype)

    # mean for inplace update
    cpu_mean = torch.full(mean_rstd_shape, float("nan"), dtype=cpu_dtype)
    npu_mean = to_ttnn(cpu_mean, device=device, dtype=npu_dtype)

    # rstd for inplace update
    cpu_rstd = torch.full(mean_rstd_shape, float("nan"), dtype=cpu_dtype)
    npu_rstd = to_ttnn(cpu_rstd, device=device, dtype=npu_dtype)

    # Forward
    npu_output, npu_mean, npu_rstd = ttnn.operations.moreh.layer_norm(
        npu_input,
        normalized_dims,
        eps,
        npu_gamma,
        npu_beta,
        output=npu_output,
        mean=npu_mean if create_mean_rstd else None,
        rstd=npu_rstd if create_mean_rstd else None,
        compute_kernel_config=compute_kernel_config,
    )

    tt_output = to_torch(npu_output, shape=input_shape)
    tt_mean = to_torch(npu_mean, shape=mean_rstd_shape) if create_mean_rstd else None
    tt_rstd = to_torch(npu_rstd, shape=mean_rstd_shape) if create_mean_rstd else None

    return tt_output, tt_mean, tt_rstd


def tt_layer_norm_backward(
    input,
    output_grad,
    *,
    normalized_dims=1,
    eps=1e-5,
    gamma=None,
    beta=None,
    dtype=ttnn.bfloat16,
    device=None,
    compute_kernel_config=None,
):
    normalized_shape = input.shape[-normalized_dims:]

    # rank
    input_shape = list(input.shape)
    input_rank = len(input_shape)

    # mean_rstd_shape
    mean_rstd_shape = input_shape[:-normalized_dims]

    # gamma_beta_shape
    gamma_beta_shape = input_shape[-normalized_dims:]

    # dtype
    cpu_dtype = torch.bfloat16
    npu_dtype = dtype

    # input
    npu_input = to_ttnn(input, device=device, dtype=npu_dtype)

    # output_grad
    npu_output_grad = to_ttnn(output_grad, device=device, dtype=npu_dtype)

    # gamma
    npu_gamma = to_ttnn(gamma, device=device, dtype=npu_dtype)

    # mean, rstd
    mean_rstd_dims = list(range(-normalized_dims, 0))

    mean = input.clone().mean(dim=mean_rstd_dims, keepdim=True)
    var = ((input.clone() - mean) ** 2).mean(dim=mean_rstd_dims, keepdim=True)
    rstd = (var + eps).rsqrt()

    npu_mean = to_ttnn(mean, device=device, dtype=npu_dtype, shape=mean_rstd_shape)
    npu_rstd = to_ttnn(rstd, device=device, dtype=npu_dtype, shape=mean_rstd_shape)

    # input_grad for inplace update
    cpu_input_grad = torch.full(input_shape, float("nan"), dtype=cpu_dtype)
    npu_input_grad = to_ttnn(cpu_input_grad, device=device, dtype=npu_dtype)

    # gamma_grad for inplace update
    npu_gamma_grad = None
    if gamma is not None:
        cpu_gamma_grad = torch.full(gamma_beta_shape, float("nan"), dtype=cpu_dtype)
        npu_gamma_grad = to_ttnn(cpu_gamma_grad, device=device, dtype=npu_dtype)

    # beta_grad for inplace update
    npu_beta_grad = None
    if beta is not None:
        cpu_beta_grad = torch.full(gamma_beta_shape, float("nan"), dtype=cpu_dtype)
        npu_beta_grad = to_ttnn(cpu_beta_grad, device=device, dtype=npu_dtype)

    # Backward
    _, npu_gamma_grad, _ = ttnn.operations.moreh.layer_norm_backward(
        npu_output_grad,
        npu_input,
        npu_mean,
        npu_rstd,
        normalized_dims,
        gamma=npu_gamma,
        input_grad=npu_input_grad,
        gamma_grad=npu_gamma_grad,
        beta_grad=npu_beta_grad,
        compute_kernel_config=compute_kernel_config,
    )

    tt_input_grad = to_torch(npu_input_grad, shape=input_shape)
    tt_gamma_grad = to_torch(npu_gamma_grad, shape=gamma_beta_shape)
    if tt_gamma_grad is not None:
        tt_gamma_grad = tt_gamma_grad.view(normalized_shape)
    tt_beta_grad = to_torch(npu_beta_grad, shape=gamma_beta_shape)
    if tt_beta_grad is not None:
        tt_beta_grad = tt_beta_grad.view(normalized_shape)

    return tt_input_grad, tt_gamma_grad, tt_beta_grad


def make_input_tensors(input_shape, normalized_dims, elementwise_affine, do_backward=False):
    # output_grad_shape
    output_grad_shape = input_shape

    # gamma_beta_shape
    gamma_beta_shape = input_shape[-normalized_dims:]

    # dtype
    cpu_dtype = torch.bfloat16

    # input
    cpu_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)

    # gamma
    cpu_gamma = None
    if elementwise_affine:
        cpu_gamma = torch.rand(gamma_beta_shape, dtype=cpu_dtype) * 2 - 1.05

    # beta
    cpu_beta = None
    if elementwise_affine:
        cpu_beta = torch.rand(gamma_beta_shape, dtype=cpu_dtype) * 2 - 1.05

    # output_grad
    cpu_output_grad = None
    if do_backward:
        cpu_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype)

    return cpu_input, cpu_gamma, cpu_beta, cpu_output_grad


def run_moreh_layer_norm(
    input_shape_normalized_dims,
    elementwise_affine,
    eps,
    dtype,
    device,
    create_mean_rstd=True,
    compute_kernel_options=None,
):
    input_shape, normalized_dims = input_shape_normalized_dims

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    cpu_input, cpu_gamma, cpu_beta, _ = make_input_tensors(input_shape, normalized_dims, elementwise_affine)

    tt_layer_norm(
        cpu_input,
        normalized_dims=normalized_dims,
        eps=eps,
        gamma=cpu_gamma,
        beta=cpu_beta,
        dtype=dtype,
        device=device,
        compute_kernel_config=compute_kernel_config,
        create_mean_rstd=create_mean_rstd,
    )


def run_moreh_layer_norm_backward(
    input_shape_normalized_dims, elementwise_affine, eps, dtype, device, compute_kernel_options=None
):
    input_shape, normalized_dims = input_shape_normalized_dims

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    cpu_input, cpu_gamma, cpu_beta, cpu_output_grad = make_input_tensors(
        input_shape, normalized_dims, elementwise_affine, do_backward=True
    )

    tt_layer_norm_backward(
        cpu_input,
        cpu_output_grad,
        normalized_dims=normalized_dims,
        eps=eps,
        gamma=cpu_gamma,
        beta=cpu_beta,
        dtype=dtype,
        device=device,
        compute_kernel_config=compute_kernel_config,
    )


@skip_for_grayskull("Using the transpose function in copy_tile causes a hang.")
@pytest.mark.parametrize("eps", [1e-5], ids=["1e-5"])
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=[
        "bfloat16",
    ],
)
@pytest.mark.parametrize(
    "elementwise_affine",
    [True],
    ids=["elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([8, 512, 768], 1),  # test 3d: GPT2-Small case
    ],
)
def test_moreh_layer_norm(input_shape_normalized_dims, elementwise_affine, eps, dtype, device):
    torch.manual_seed(2023)
    run_moreh_layer_norm(input_shape_normalized_dims, elementwise_affine, eps, dtype, device)


@skip_for_blackhole("Mismatching on BH, see #12349")
@skip_for_grayskull("Using the transpose function in copy_tile causes a hang.")
@pytest.mark.parametrize("eps", [1e-5], ids=["1e-5"])
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=[
        "bfloat16",
    ],
)
@pytest.mark.parametrize(
    "elementwise_affine",
    [True],
    ids=["elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        # ([8, 512, 768], 1),  # test 3d: GPT2-Small case
        ([8, 254, 257], 1),  # test 3d: GPT2-Small case
    ],
)
def test_moreh_layer_norm_backward(input_shape_normalized_dims, elementwise_affine, eps, dtype, device):
    torch.manual_seed(2023)
    run_moreh_layer_norm_backward(input_shape_normalized_dims, elementwise_affine, eps, dtype, device)
