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
