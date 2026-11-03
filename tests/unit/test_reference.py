"""Cache protocol and mathematical attention behavior, with no NPU imports."""
from dataclasses import replace
import math

import pytest
import torch

from npu_sparse_attn_sharedkv_int8 import CacheQuantization
from npu_sparse_attn_sharedkv_int8.reference import (
    joint_quantized_row, make_case, sharedkv_reference, tiled_quantized_reference,
)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_reject_invalid_frozen_scales(bad):
    scale = torch.ones(512)
    scale[17] = bad
    with pytest.raises(ValueError):
        CacheQuantization(scale)


def test_symmetric_rounding_and_clipping():
    quant = CacheQuantization(torch.ones(512))
    row = torch.zeros(1, 512)
    row[0, :8] = torch.tensor([-200.0, -127.0, -1.5, -0.5, 0.5, 1.5, 127.0, 200.0])
    assert quant.quantize(row)[0, :8].tolist() == [-127, -127, -2, 0, 0, 2, 127, 127]
    assert quant.quantize(row).dtype == torch.int8


def test_appending_tokens_does_not_change_cache_scale_or_old_bytes():
    quant = CacheQuantization.calibrate([torch.ones(3, 512)])
    scale = quant.scale.clone()
    inverse = quant.inverse_scale.clone()
    prefix = torch.linspace(-1, 1, 512).reshape(1, 512)
    before = quant.quantize(prefix)
    quant.quantize(torch.full((2, 512), 1000.0))
    assert torch.equal(quant.scale, scale)
    assert torch.equal(quant.inverse_scale, inverse)
    assert torch.equal(quant.quantize(prefix), before)


def test_msmodelslim_requires_zero_offsets():
    tensors = {"layer.kv_cache_scale": torch.ones(512), "layer.kv_cache_offset": torch.zeros(512)}
    assert torch.equal(CacheQuantization.from_msmodelslim(tensors, "layer").scale, torch.ones(512))
    tensors["layer.kv_cache_offset"][0] = 1
    with pytest.raises(ValueError, match="nonzero offsets"):
        CacheQuantization.from_msmodelslim(tensors, "layer")


def test_original_compressed_and_sink_share_one_denominator():
    quant = CacheQuantization(torch.ones(512))
    q = torch.zeros(2, 512)
    caches = [torch.full((1, 512), value, dtype=torch.int8) for value in (10, 20)]
    out, lse = joint_quantized_row(q, caches, [quant, quant], torch.zeros(2), 1.0)
    torch.testing.assert_close(out.float(), torch.full_like(q, 10.0), rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.full((2,), math.log(3)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_integer_residual_oracle_against_fp64_dequantized_attention(ratio):
    case = make_case(query_lengths=(2, 1), seq_lens=(129, 257), ratio=ratio, seed=23)
    oq = CacheQuantization.calibrate([case.ori])
    cq = CacheQuantization.calibrate([case.cmp]) if case.cmp is not None else None
    quantized = replace(case, ori=oq.quantize(case.ori), cmp=cq.quantize(case.cmp) if cq else None)
    decoded = replace(quantized, ori=oq.dequantize(quantized.ori),
                      cmp=cq.dequantize(quantized.cmp) if cq else None)
    expected, expected_lse = sharedkv_reference(decoded)
    actual, lse = tiled_quantized_reference(quantized, oq, cq, return_lse=True)
    relative_l2 = (actual.float() - expected).norm() / expected.norm()
    assert float(relative_l2) < 0.01
    torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-4)


def test_sparse_padding_indices_do_not_contribute():
    case = make_case(query_lengths=(1,), seq_lens=(257,), ratio=4, seed=9)
    expected = sharedkv_reference(case)
    indices = case.cmp_indices.clone()
    indices[indices < 0] = 2**30
    padded = replace(case, cmp_indices=indices)
    actual = sharedkv_reference(padded)
    for result, golden in zip(actual, expected):
        torch.testing.assert_close(result, golden, rtol=0, atol=0)
