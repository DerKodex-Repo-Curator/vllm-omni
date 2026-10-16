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


def _qwen3_attention_fused_forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """``Qwen3Attention.forward`` with Q/K RMSNorm and RoPE as one in-place kernel."""
    qkv, _ = self.qkv_proj(hidden_states)
    torch.ops._C.fused_qk_norm_rope(
        qkv,
        self.num_heads,
        self.num_kv_heads,
        self.num_kv_heads,
        self.head_dim,
        self.q_norm.variance_epsilon,
        self.q_norm.weight,
        self.k_norm.weight,
        self.rotary_emb.cos_sin_cache,
        self.rotary_emb.is_neox_style,
        positions,
    )
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    output, _ = self.o_proj(self.attn(q, k, v))
    return output


def fuse_qwen3_qk_norm_rope(model: torch.nn.Module, vllm_config) -> int:
    """Route every Qwen3 attention of ``model`` through vLLM's fused Q/K norm + RoPE op.

    vLLM applies this op only as a compile pass; without compilation the
    three ops run separately. Only for an uncompiled CUDA BF16/FP16 model.
    """
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3 import Qwen3Attention

    if (
        vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
        or not current_platform.is_cuda()
        or vllm_config.model_config.dtype not in (torch.bfloat16, torch.float16)
        or not hasattr(torch.ops._C, "fused_qk_norm_rope")
    ):
        return 0
    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention) and getattr(module.rotary_emb, "cos_sin_cache", None) is not None:
            module.forward = _qwen3_attention_fused_forward.__get__(module)
            count += 1
    return count
