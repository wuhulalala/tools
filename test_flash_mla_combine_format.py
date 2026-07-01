#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
import triton

from flaggems_vllm.ops.flash_mla import (
    FLASH_MLA_COMBINE_BLOCK_D,
    FLASH_MLA_COMBINE_BLOCK_H,
    flash_mla_combine_kernel_compact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--h-q", type=int, default=128)
    parser.add_argument("--dv", type=int, default=512)
    parser.add_argument("--splits", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    total_splits = args.batch * args.splits
    out_accum = torch.randn(
        (total_splits, args.h_q, args.dv),
        dtype=dtype,
        device=device,
    )
    lse_accum = torch.randn(
        (total_splits, args.h_q),
        dtype=torch.float32,
        device=device,
    )
    num_splits = torch.arange(
        0,
        total_splits + 1,
        args.splits,
        dtype=torch.int32,
        device=device,
    )
    combine_req_ids = torch.arange(args.batch, dtype=torch.int32, device=device)
    num_combine_reqs = torch.tensor([args.batch], dtype=torch.int32, device=device)
    out = torch.empty((args.batch, args.h_q, args.dv), dtype=dtype, device=device)

    flash_mla_combine_kernel_compact[
        (
            args.batch,
            triton.cdiv(args.h_q, FLASH_MLA_COMBINE_BLOCK_H),
            triton.cdiv(args.dv, FLASH_MLA_COMBINE_BLOCK_D),
        )
    ](
        out_accum,
        lse_accum,
        num_splits,
        combine_req_ids,
        num_combine_reqs,
        out,
        args.h_q,
        out_accum.stride(0),
        out_accum.stride(1),
        lse_accum.stride(0),
        lse_accum.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=FLASH_MLA_COMBINE_BLOCK_H,
        BLOCK_D=FLASH_MLA_COMBINE_BLOCK_D,
        HEAD_DIM_V=args.dv,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    ref = torch.empty_like(out, dtype=torch.float32)
    for b in range(args.batch):
        start = b * args.splits
        end = start + args.splits
        lse = lse_accum[start:end].float()
        max_lse = torch.max(lse, dim=0).values
        weights = torch.exp(lse - max_lse)
        denom = torch.sum(weights, dim=0)
        acc = torch.sum(weights[:, :, None] * out_accum[start:end].float(), dim=0)
        ref[b] = acc / denom[:, None]

    diff = (out.float() - ref).abs()
    print(
        f"batch={args.batch} splits={args.splits} h_q={args.h_q} dv={args.dv} "
        f"max_abs={diff.max().item():.6f} mean_abs={diff.mean().item():.6f} "
        f"allclose_2e-2={torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)}"
    )


if __name__ == "__main__":
    main()

