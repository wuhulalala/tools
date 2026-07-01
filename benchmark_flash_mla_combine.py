#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os

import torch
import triton

from benchmark_flash_mla import make_inputs
from flaggems_vllm.ops.flash_mla import (
    FLASH_MLA_COMBINE_BLOCK_D,
    FLASH_MLA_COMBINE_BLOCK_H,
    flash_mla_combine_kernel_compact,
    get_flash_mla_tle_decode_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--s-kv", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--tle-variant", choices=("auto", "qtail_global"), default="auto")
    args = parser.parse_args()

    os.environ.pop("FLAGGEMS_VLLM_FLASH_MLA_TLE", None)
    os.environ["FLAGGEMS_VLLM_FLASH_MLA_TLE_VARIANT"] = args.tle_variant

    inputs = make_inputs(args.s_kv, args.batch, 1)
    (
        q,
        block_table,
        blocked_k,
        _max_seqlen_pad,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    ) = inputs
    plan = get_flash_mla_tle_decode_plan(
        b=b,
        s_q=s_q,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        dv=dv,
        block_size=block_size,
        dtype=q.dtype,
        device=q.device,
        causal=causal,
        reuse_output=True,
    )
    plan.plan(cache_seqlens)
    out = torch.empty((b * s_q, h_q, dv), dtype=q.dtype, device=q.device)
    plan.run(
        q=q,
        blocked_k=blocked_k,
        block_table=block_table,
        update_metadata=False,
        out=out,
    )
    torch.cuda.synchronize()

    def combine_only():
        flash_mla_combine_kernel_compact[
            (
                plan.max_combine_reqs,
                triton.cdiv(plan.h_q, FLASH_MLA_COMBINE_BLOCK_H),
                triton.cdiv(plan.dv, FLASH_MLA_COMBINE_BLOCK_D),
            )
        ](
            plan.out_accum,
            plan.lse_accum,
            plan.num_splits,
            plan.combine_req_ids,
            plan.num_combine_reqs,
            out,
            plan.h_q,
            plan.out_accum.stride(0),
            plan.out_accum.stride(1),
            plan.lse_accum.stride(0),
            plan.lse_accum.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_H=FLASH_MLA_COMBINE_BLOCK_H,
            BLOCK_D=FLASH_MLA_COMBINE_BLOCK_D,
            HEAD_DIM_V=plan.dv,
            num_warps=4,
            num_stages=1,
        )

    combine_only()
    torch.cuda.synchronize()
    latency_ms = triton.testing.do_bench(
        combine_only,
        warmup=args.warmup,
        rep=args.iter,
        return_mode="median",
    )
    num_combine_reqs = int(plan.num_combine_reqs.cpu().item())
    print(
        f"batch={args.batch} s_kv={args.s_kv} variant={args.tle_variant} "
        f"num_sm_parts={plan.num_sm_parts} max_combine_reqs={plan.max_combine_reqs} "
        f"num_combine_reqs={num_combine_reqs} combine_ms={latency_ms:.4f}"
    )


if __name__ == "__main__":
    main()

