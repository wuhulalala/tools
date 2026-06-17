#!/usr/bin/env python3
"""Correctness test: flash_mla across all batch/s_kv/h_q combos (d=576, dv=512 fixed).

Usage:
  cd /workspace/FlagGems-vllm
  PYTHONPATH=/workspace/FlagGems-vllm/src \
  FLAGGEMS_VLLM_FLASH_MLA_TLE=1 \
  FLAGGEMS_VLLM_FLASH_MLA_TLE_VARIANT=auto \
  /workspace/gemms_env/bin/python -m pytest -q tools/smoke_test_flash_mla_shapes.py -s
"""

import math
import pytest
import torch
import triton
import flaggems_vllm

device = flaggems_vllm.device


try:
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache as vllm_flash_mla_with_kvcache,
        get_mla_metadata as vllm_get_mla_metadata,
    )

    HAS_VLLM_FLASH_MLA = True
except ImportError:
    HAS_VLLM_FLASH_MLA = False


def vllm_flash_mla(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal,
):
    _ = (max_seqlen_pad, block_size, b, s_q, h_q, h_kv, d)
    sched_meta, _ = vllm_get_mla_metadata()
    out, _ = vllm_flash_mla_with_kvcache(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        dv,
        sched_meta,
        None,
        softmax_scale=1 / math.sqrt(q.shape[-1]),
        causal=causal,
    )
    return out

def cal_diff(x, y, name):
    x, y = x.double(), y.double()
    x = x.to(y.device)
    RMSE = ((x - y) ** 2).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    assert cos_diff < 1e-5, f"{name}: cos_diff={cos_diff:.2e} RMSE={RMSE:.2e} amax_diff={amax_diff:.2e}"


def ref_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q,
            cache_seqlens, h_q, h_kv, d, dv, causal):
    def _sdpa(query, key, value, h_q, h_kv, is_causal=False):
        query, key, value = query.float(), key.float(), value.float()
        key = key.repeat_interleave(h_q // h_kv, dim=0)
        value = value.repeat_interleave(h_q // h_kv, dim=0)
        scale = 1.0 / math.sqrt(query.size(-1))
        attn_weight = query @ key.transpose(-2, -1) * scale
        if is_causal:
            sq, sk = query.shape[-2], key.shape[-2]
            bias = torch.zeros(sq, sk, dtype=query.dtype, device=query.device)
            bias.masked_fill_(
                torch.ones(sq, sk, dtype=torch.bool, device=query.device)
                .tril(diagonal=sk - sq).logical_not(), float("-inf"))
            attn_weight += bias
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        return attn_weight @ value

    blocked_v = blocked_k[..., :dv]
    out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=torch.device("cpu"))
    bk_cpu, bv_cpu = blocked_k.cpu(), blocked_k.cpu()[..., :dv]
    for i in range(b):
        begin = i * max_seqlen_pad
        end = begin + cache_seqlens[i].item()
        O = _sdpa(
            q[i].cpu().transpose(0, 1),
            bk_cpu.view(-1, h_kv, d)[begin:end].transpose(0, 1),
            bv_cpu.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
            h_q=h_q, h_kv=h_kv, is_causal=causal,
        )
        out[i] = O.transpose(0, 1)
    return out.to(device=q.device, dtype=q.dtype)


BATCH_LIST = [1, 2, 4, 8, 16]
S_KV_LIST = [4096]
SHAPE_GROUPS = [
    {"h_q": 128, "h_kv": 1, "d_qk": 576, "dv": 512},
    {"h_q": 64,  "h_kv": 1, "d_qk": 576, "dv": 512},
]

ALL_COMBOS = [
    (batch, s_kv) for batch in BATCH_LIST for s_kv in S_KV_LIST
]


def make_inputs(seqlen, batch, h_q, h_kv, d, dv):
    s_q = 1
    block_size = 64
    dtype = torch.bfloat16
    torch.manual_seed(42)
    cache_seqlens = torch.tensor(
        [seqlen + 2 * i for i in range(batch)], dtype=torch.int32, device=device)
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    q = torch.randn([batch, s_q, h_q, d], dtype=dtype, device=device)
    block_table = torch.arange(
        batch * max_seqlen_pad // block_size, dtype=torch.int32, device=device
    ).view(batch, max_seqlen_pad // block_size)
    blocked_k = torch.randn(
        [block_table.numel(), block_size, h_kv, d], dtype=dtype, device=device)
    causal = True
    return q, block_table, blocked_k, max_seqlen_pad, block_size, batch, s_q, cache_seqlens, h_q, h_kv, d, dv, causal


@pytest.mark.parametrize("shape_group", SHAPE_GROUPS, ids=["h128_d576", "h64_d576"])
@pytest.mark.parametrize("batch,s_kv", ALL_COMBOS)
def test_flash_mla_shapes(shape_group, batch, s_kv):
    h_q, h_kv, d, dv = shape_group["h_q"], shape_group["h_kv"], shape_group["d_qk"], shape_group["dv"]
    inputs = make_inputs(s_kv, batch, h_q, h_kv, d, dv)
    ref_out = ref_mla(*inputs)
    res_out = flaggems_vllm.flash_mla(*inputs)
    cal_diff(res_out.float(), ref_out.float(),
             f"h_q={h_q} batch={batch} s_kv={s_kv}")
