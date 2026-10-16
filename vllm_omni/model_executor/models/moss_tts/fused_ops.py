# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Explicit fused kernels for the MOSS-TTS Local codec and Local predictor.

These replace the fusions Inductor used to generate, so neither stage needs
``torch.compile``:

- ``rope_qk_``: the codec's GPT-J (interleaved) rotation of Q and K, in place
  on the QKV projection, from a shared per-stage cos/sin table. Eager PyTorch
  spends about 20 elementwise kernels per layer on it.
- ``add_layer_norm``: ``residual + update * scale`` (layer scale), or
  ``residual + update``, followed by LayerNorm, in one pass. It returns both
  the new residual and the normalized activations, so one launch covers a
  residual add and the next norm, also across a layer boundary.

Arithmetic is FP32 with one rounding to the storage dtype where the eager ops
round, so results differ from eager PyTorch only at rounding level (FMA
contraction and the LayerNorm reduction order).
"""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_gdc_launch = tl.extra.cuda.gdc_launch_dependents
_gdc_wait = tl.extra.cuda.gdc_wait
_USE_PDL = current_platform.is_cuda() and current_platform.has_device_capability(90)


@triton.jit
def _rope_qk(
    qkv_ptr,
    cos_ptr,
    sin_ptr,
    T,
    s_b,
    s_t,
    s_c,
    s_h,
    c_b,
    c_t,
    H: tl.constexpr,
    BH: tl.constexpr,
    HALF: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    pid = tl.program_id(0)
    b = pid // T
    t = pid - b * T
    i = tl.arange(0, HALF)
    cos = tl.load(cos_ptr + b * c_b + t * c_t + i)
    sin = tl.load(sin_ptr + b * c_b + t * c_t + i)
    h = tl.arange(0, BH)
    mask = (h < H)[:, None]
    base = qkv_ptr + b.to(tl.int64) * s_b + t.to(tl.int64) * s_t + h[:, None] * s_h + 2 * i[None, :]
    for c in tl.static_range(2):
        ptr = base + c * s_c
        re = tl.load(ptr, mask).to(tl.float32)
        im = tl.load(ptr + 1, mask).to(tl.float32)
        out_re = re * cos[None, :] - im * sin[None, :]
        out_im = re * sin[None, :] + im * cos[None, :]
        tl.store(ptr, out_re.to(qkv_ptr.dtype.element_ty), mask)
        tl.store(ptr + 1, out_im.to(qkv_ptr.dtype.element_ty), mask)


def rope_qk_(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """Rotate Q and K of ``qkv`` (B, T, 3, H, D) in place, GPT-J pairs.

    ``cos``/``sin`` are FP32 (B, 1, T, D/2) as ``rope_angles`` returns them.
    """
    B, T, three, H, D = qkv.shape
    assert three == 3 and qkv.stride(4) == 1 and D % 2 == 0
    assert cos.shape == (B, 1, T, D // 2) and cos.stride(3) == 1 and sin.stride() == cos.stride()
    if B * T == 0:
        return
    _rope_qk[(B * T,)](
        qkv,
        cos,
        sin,
        T,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        cos.stride(0),
        cos.stride(2),
        H,
        triton.next_power_of_2(H),
        D // 2,
        USE_PDL=_USE_PDL,
        launch_pdl=_USE_PDL,
        num_warps=4,
    )


@triton.jit
def _add_layer_norm(
    x_ptr,
    u_ptr,
    scale_ptr,
    w_ptr,
    b_ptr,
    res_ptr,
    out_ptr,
    N: tl.constexpr,
    eps,
    BN: tl.constexpr,
    HAS_UPDATE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BN)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, m, 0.0).to(tl.float32)
    if HAS_UPDATE:
        u = tl.load(u_ptr + row * N + cols, m, 0.0).to(tl.float32)
        if HAS_SCALE:
            u = u * tl.load(scale_ptr + cols, m, 0.0).to(tl.float32)
        # one rounding to the residual dtype, as torch.addcmul / add store it
        r = (x + u).to(res_ptr.dtype.element_ty)
        tl.store(res_ptr + row * N + cols, r, m)
        x = r.to(tl.float32)
    mean = tl.sum(x, 0) / N
    xc = tl.where(m, x - mean, 0.0)
    var = tl.sum(xc * xc, 0) / N
    y = xc * tl.rsqrt(var + eps) * tl.load(w_ptr + cols, m, 0.0).to(tl.float32)
    if HAS_BIAS:
        y = y + tl.load(b_ptr + cols, m, 0.0).to(tl.float32)
    tl.store(out_ptr + row * N + cols, y.to(out_ptr.dtype.element_ty), m)


def add_layer_norm(
    x: torch.Tensor,
    update: torch.Tensor | None,
    scale: torch.Tensor | None,
    norm: torch.nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(r, layer_norm(r))`` with ``r = x + update * scale`` (``x`` if no update).

    All tensors share the last dimension; ``x``/``update`` must be contiguous.
    """
    N = x.shape[-1]
    assert x.is_contiguous() and (update is None or (update.is_contiguous() and update.shape == x.shape))
    assert norm.weight is not None and norm.weight.shape == (N,)
    rows = x.numel() // N
    res = torch.empty_like(x) if update is not None else x
    out = torch.empty_like(x)
    if rows:
        _add_layer_norm[(rows,)](
            x,
            update if update is not None else x,
            scale if scale is not None else norm.weight,
            norm.weight,
            norm.bias if norm.bias is not None else norm.weight,
            res,
            out,
            N,
            float(norm.eps),
            triton.next_power_of_2(N),
            update is not None,
            scale is not None,
            norm.bias is not None,
            USE_PDL=_USE_PDL,
            launch_pdl=_USE_PDL,
            num_warps=4 if N <= 2048 else 8,
        )
    return res, out


@triton.jit
def _rms_norm(
    x_ptr,
    r_ptr,
    w_ptr,
    out_ptr,
    res_ptr,
    sx,
    sr,
    N: tl.constexpr,
    eps,
    BN: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BN)
    m = cols < N
    x = tl.load(x_ptr + row * sx + cols, m, 0.0).to(tl.float32)
    if HAS_RESIDUAL:
        x = x + tl.load(r_ptr + row * sr + cols, m, 0.0).to(tl.float32)
        tl.store(res_ptr + row * N + cols, x.to(res_ptr.dtype.element_ty), m)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / N + eps)
    # as vLLM's rms_norm: normalize in FP32, round, then scale by the weight
    y = (x * rstd).to(out_ptr.dtype.element_ty).to(tl.float32) * tl.load(w_ptr + cols, m, 0.0).to(tl.float32)
    tl.store(out_ptr + row * N + cols, y.to(out_ptr.dtype.element_ty), m)


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """vLLM ``RMSNorm`` (``fused_add_rms_norm`` with ``residual``) as one PDL-launched kernel."""
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    rows = x2.shape[0]
    out = torch.empty((rows, N), dtype=x.dtype, device=x.device)
    res = None
    if residual is not None:
        r2 = residual.reshape(-1, N)
        res = torch.empty_like(out)
    if rows:
        _rms_norm[(rows,)](
            x2,
            r2 if residual is not None else x2,
            weight,
            out,
            res if res is not None else out,
            x2.stride(0),
            r2.stride(0) if residual is not None else 0,
            N,
            eps,
            triton.next_power_of_2(N),
            residual is not None,
            USE_PDL=_USE_PDL,
            launch_pdl=_USE_PDL,
            num_warps=4 if N <= 4096 else 8,
        )
    if residual is None:
        return out.view(x.shape)
    return out.view(x.shape), res.view(residual.shape)


@triton.jit
def _silu_and_mul(
    x_ptr,
    out_ptr,
    sx,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < D
    g = tl.load(x_ptr + row * sx + cols, m, 0.0).to(tl.float32)
    u = tl.load(x_ptr + row * sx + D + cols, m, 0.0).to(tl.float32)
    # as F.silu(gate) * up on BF16 tensors: two roundings
    act = (g / (1.0 + tl.exp(-g))).to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + row * D + cols, (act * u).to(out_ptr.dtype.element_ty), m)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :d]) * x[..., d:]`` as one PDL-launched kernel."""
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, x.shape[-1])
    out = torch.empty((x2.shape[0], d), dtype=x.dtype, device=x.device)
    if x2.shape[0]:
        block = 1024
        _silu_and_mul[(x2.shape[0], triton.cdiv(d, block))](
            x2, out, x2.stride(0), d, block, USE_PDL=_USE_PDL, launch_pdl=_USE_PDL, num_warps=4
        )
    return out.view(*x.shape[:-1], d)


@triton.jit
def _qk_norm_rope(
    qkv_ptr,
    pos_ptr,
    qw_ptr,
    kw_ptr,
    cache_ptr,
    sqkv,
    scache,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    eps,
    BH: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    t = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    hmask = (heads < HQ + HK)[:, None]
    half: tl.constexpr = D // 2
    i = tl.arange(0, half)
    pos = tl.load(pos_ptr + t)
    cos = tl.load(cache_ptr + pos * scache + i).to(tl.float32)[None, :]
    sin = tl.load(cache_ptr + pos * scache + half + i).to(tl.float32)[None, :]
    base = qkv_ptr + t * sqkv + heads[:, None] * D + i[None, :]
    x1 = tl.load(base, hmask, 0.0).to(tl.float32)
    x2 = tl.load(base + half, hmask, 0.0).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / D + eps)[:, None]
    is_q = (heads < HQ)[:, None]
    w1 = tl.where(is_q, tl.load(qw_ptr + i)[None, :], tl.load(kw_ptr + i)[None, :]).to(tl.float32)
    w2 = tl.where(is_q, tl.load(qw_ptr + half + i)[None, :], tl.load(kw_ptr + half + i)[None, :]).to(tl.float32)
    dtype = qkv_ptr.dtype.element_ty
    # vLLM rms_norm per head (normalize, round, scale, round), then NeoX RoPE
    n1 = ((x1 * rstd).to(dtype).to(tl.float32) * w1).to(dtype).to(tl.float32)
    n2 = ((x2 * rstd).to(dtype).to(tl.float32) * w2).to(dtype).to(tl.float32)
    tl.store(base, (n1 * cos - n2 * sin).to(dtype), hmask)
    tl.store(base + half, (n2 * cos + n1 * sin).to(dtype), hmask)


def qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    """Per-head RMSNorm of Q and K, then NeoX RoPE, in place on ``qkv`` (T, (Hq + 2 Hk) * D)."""
    tokens = qkv.shape[0]
    if tokens:
        heads = num_heads + num_kv_heads
        bh = 8
        _qk_norm_rope[(tokens, triton.cdiv(heads, bh))](
            qkv,
            positions,
            q_weight,
            k_weight,
            cos_sin_cache,
            qkv.stride(0),
            cos_sin_cache.stride(0),
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            bh,
            USE_PDL=_USE_PDL,
            launch_pdl=_USE_PDL,
            num_warps=4,
        )


def _qwen3_attention_fused_forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """``Qwen3Attention.forward`` with Q/K RMSNorm and RoPE as one in-place kernel."""
    qkv, _ = self.qkv_proj(hidden_states)
    qk_norm_rope_(
        qkv,
        positions,
        self.num_heads,
        self.num_kv_heads,
        self.head_dim,
        self.q_norm.variance_epsilon,
        self.q_norm.weight,
        self.k_norm.weight,
        self.rotary_emb.cos_sin_cache,
    )
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    output, _ = self.o_proj(self.attn(q, k, v))
    return output


def _rms_norm_forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
    return rms_norm(x, self.weight, self.variance_epsilon, residual)


def _silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    return silu_and_mul(x)


def fuse_qwen3_backbone(model: torch.nn.Module, vllm_config) -> dict[str, int]:
    """Route an uncompiled Qwen3 backbone through the PDL-launched kernels above.

    Without compilation vLLM runs its CUDA custom ops for RMSNorm, SiLU-and-mul
    and the Q/K norm + RoPE; at decode sizes they are slower than the kernels
    Inductor generates. Only for an uncompiled CUDA BF16/FP16 model; the
    attention, GEMMs and KV cache stay vLLM's.
    """
    from vllm.config import CompilationMode
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.qwen3 import Qwen3Attention

    counts = {"attention": 0, "rms_norm": 0, "silu_and_mul": 0}
    if (
        vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
        or not current_platform.is_cuda()
        or vllm_config.model_config.dtype not in (torch.bfloat16, torch.float16)
    ):
        return counts
    for module in model.modules():
        rope = getattr(module, "rotary_emb", None)
        if (
            isinstance(module, Qwen3Attention)
            and getattr(rope, "is_neox_style", False)
            and getattr(rope, "rotary_dim", module.head_dim) == module.head_dim
            and getattr(rope, "cos_sin_cache", None) is not None
        ):
            module.forward = _qwen3_attention_fused_forward.__get__(module)
            counts["attention"] += 1
        elif type(module) is RMSNorm and module.has_weight and module.variance_size_override is None:
            module.forward = _rms_norm_forward.__get__(module)
            counts["rms_norm"] += 1
        elif type(module) is SiluAndMul:
            module.forward = _silu_and_mul_forward.__get__(module)
            counts["silu_and_mul"] += 1
    return counts
