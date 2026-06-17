#!/usr/bin/env python3
"""Run one FlashMLA invocation under CUDA profiler control.

Usage examples:
  FLAGGEMS_VLLM_FLASH_MLA_TLE=1 python tools/profile_flash_mla_single.py --impl gems --seqlen 8192
  python tools/profile_flash_mla_single.py --impl vllm --seqlen 8192
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import triton

import flaggems_vllm


def make_inputs(seqlen: int, batch: int = 128):
    device = flaggems_vllm.device
    dtype = torch.bfloat16
    s_q = 1
    h_q = 128
    h_kv = 1
    d = 576
    dv = 512
    block_size = 64
    causal = True
    torch.manual_seed(0)
    cache_seqlens = torch.tensor(
        [seqlen + 2 * i for i in range(batch)], dtype=torch.int32, device=device
    )
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    q = torch.randn([batch, s_q, h_q, d], dtype=dtype, device=device)
    block_table = torch.arange(
        batch * max_seqlen_pad // block_size, dtype=torch.int32, device=device
    ).view(batch, max_seqlen_pad // block_size)
    blocked_k = torch.randn(
        [block_table.numel(), block_size, h_kv, d], dtype=dtype, device=device
    )
    return (
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        block_size,
        batch,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    )


def run_gems(args_tuple):
    return flaggems_vllm.flash_mla(*args_tuple)


def run_vllm(args_tuple):
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    (
        q,
        block_table,
        blocked_k,
        _max_seqlen_pad,
        _block_size,
        _batch,
        _s_q,
        cache_seqlens,
        _h_q,
        _h_kv,
        d,
        dv,
        causal,
    ) = args_tuple
    sched_meta, _ = get_mla_metadata()
    out, _ = flash_mla_with_kvcache(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        dv,
        sched_meta,
        None,
        softmax_scale=1 / math.sqrt(d),
        causal=causal,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("gems", "vllm"), default="gems")
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cuda-profiler-api", action="store_true")
    ns = parser.parse_args()

    inputs = make_inputs(ns.seqlen, ns.batch)
    fn = run_gems if ns.impl == "gems" else run_vllm
    for _ in range(ns.warmup):
        out = fn(inputs)
    torch.cuda.synchronize()

    if ns.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()
    out = fn(inputs)
    torch.cuda.synchronize()
    if ns.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()
    print(
        f"impl={ns.impl} tle={os.environ.get('FLAGGEMS_VLLM_FLASH_MLA_TLE', '0')} "
        f"shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()}"
    )


if __name__ == "__main__":
    main()
