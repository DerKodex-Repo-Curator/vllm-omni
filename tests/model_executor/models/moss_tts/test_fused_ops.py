# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit fused kernels that replace the Inductor fusions of the MOSS codec and Local predictor."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("heads", [12, 20])
@pytest.mark.parametrize("frames", [1, 15, 240])
def test_rope_qk_matches_eager_rotation(heads, frames):
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import apply_rope, rope_angles
    from vllm_omni.model_executor.models.moss_tts.fused_ops import rope_qk_

    torch.manual_seed(0)
    B, D = 3, 64
    qkv = torch.randn(B, frames, 3, heads, D, device="cuda", dtype=torch.bfloat16)
    offset = torch.tensor([0, 7, 1234], device="cuda")
    cos, sin = rope_angles(offset, frames, D, 10_000.0)
    view = qkv.permute(2, 0, 3, 1, 4)
    q_ref, k_ref = apply_rope(view[0], view[1], offset, table=(cos, sin))
    v_ref = view[2].clone()

    rope_qk_(qkv, cos, sin)
    view = qkv.permute(2, 0, 3, 1, 4)
    # FP32 rotation rounded once to BF16; FMA contraction may move the last bit.
    torch.testing.assert_close(view[0], q_ref, atol=0, rtol=2**-7)
    torch.testing.assert_close(view[1], k_ref, atol=0, rtol=2**-7)
    assert torch.equal(view[2], v_ref)


@pytest.mark.parametrize("width", [768, 1280, 2560])
@pytest.mark.parametrize("scaled", [True, False])
def test_add_layer_norm_matches_eager(width, scaled):
    from vllm_omni.model_executor.models.moss_tts.fused_ops import add_layer_norm

    torch.manual_seed(0)
    norm = torch.nn.LayerNorm(width, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.uniform_(-0.1, 0.1)
    x = torch.randn(4, 33, width, device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(x)
    scale = torch.rand(width, device="cuda", dtype=torch.bfloat16) * 0.1 if scaled else None

    residual_ref = torch.addcmul(x, update, scale) if scaled else x + update
    out_ref = norm(residual_ref)
    residual, out = add_layer_norm(x, update, scale, norm)
    torch.testing.assert_close(residual, residual_ref, atol=0, rtol=2**-7)
    torch.testing.assert_close(out, out_ref, atol=2e-2, rtol=2e-2)

    same, normed = add_layer_norm(x, None, None, norm)
    assert same is x
    torch.testing.assert_close(normed, norm(x), atol=2e-2, rtol=2e-2)


def test_fused_layer_loop_matches_per_layer_loop():
    """The decoder transformer gives the same output with and without the fused residual/norm loop."""
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import MossAudioTokenizerTransformer

    torch.manual_seed(0)
    model = MossAudioTokenizerTransformer(
        d_model=256,
        num_heads=4,
        num_layers=3,
        dim_feedforward=1024,
        causal=True,
        context=64,
        positional_embedding="rope",
        norm="layer_norm",
        layer_scale=0.1,
        gating="none",
    ).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        fused = model(x)
        model._fused_norm_ready = lambda *_: False
        reference = model(x)
    torch.testing.assert_close(fused, reference, atol=5e-2, rtol=5e-2)


def _rms_norm_reference(x, weight, eps, residual=None):
    # vllm.ir.ops.rms_norm / fused_add_rms_norm
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
        residual = xf.to(x.dtype)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = (xf.to(weight.dtype) * weight).to(x.dtype)
    return out if residual is None else (out, residual)


@pytest.mark.parametrize("tokens", [1, 64, 300])
@pytest.mark.parametrize("with_residual", [False, True])
def test_rms_norm_matches_vllm_reference(tokens, with_residual):
    from vllm_omni.model_executor.models.moss_tts.fused_ops import rms_norm

    torch.manual_seed(0)
    x = torch.randn(tokens, 2560, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    weight = torch.rand(2560, device="cuda", dtype=torch.bfloat16) + 0.5
    got = rms_norm(x, weight, 1e-6, residual)
    ref = _rms_norm_reference(x, weight, 1e-6, residual)
    if with_residual:
        torch.testing.assert_close(got[1], ref[1], atol=0, rtol=0)
        got, ref = got[0], ref[0]
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("tokens", [1, 64, 300])
def test_silu_and_mul_matches_reference(tokens):
    import torch.nn.functional as F

    from vllm_omni.model_executor.models.moss_tts.fused_ops import silu_and_mul

    torch.manual_seed(0)
    x = torch.randn(tokens, 2 * 9728, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(silu_and_mul(x), F.silu(x[:, :9728]) * x[:, 9728:], atol=0, rtol=2**-7)


@pytest.mark.parametrize("tokens", [1, 64, 300])
def test_qk_norm_rope_matches_reference(tokens):
    from vllm_omni.model_executor.models.moss_tts.fused_ops import qk_norm_rope_

    torch.manual_seed(0)
    hq, hk, d = 32, 8, 128
    qkv = torch.randn(tokens, (hq + 2 * hk) * d, device="cuda", dtype=torch.bfloat16)
    qw = torch.rand(d, device="cuda", dtype=torch.bfloat16) + 0.5
    kw = torch.rand(d, device="cuda", dtype=torch.bfloat16) + 0.5
    positions = torch.randint(0, 4096, (tokens,), device="cuda")
    inv_freq = 1.0 / (1_000_000 ** (torch.arange(0, d, 2, device="cuda", dtype=torch.float32) / d))
    freqs = torch.arange(4096, device="cuda", dtype=torch.float32)[:, None] * inv_freq[None, :]
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)

    q, k, v = qkv.split([hq * d, hk * d, hk * d], -1)
    qn = _rms_norm_reference(q.reshape(tokens, hq, d), qw, 1e-6)
    kn = _rms_norm_reference(k.reshape(tokens, hk, d), kw, 1e-6)
    cos, sin = cache[positions].float().chunk(2, dim=-1)

    def rope(x):
        x1, x2 = x.float().chunk(2, dim=-1)
        c, s = cos[:, None], sin[:, None]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)

    out = qkv.clone()
    qk_norm_rope_(out, positions, hq, hk, d, 1e-6, qw, kw, cache)
    fq, fk, fv = out.split([hq * d, hk * d, hk * d], -1)
    torch.testing.assert_close(fq.reshape(tokens, hq, d), rope(qn), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fk.reshape(tokens, hk, d), rope(kn), atol=2e-2, rtol=2e-2)
    assert torch.equal(fv, v)
