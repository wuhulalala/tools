#!/usr/bin/env python3
"""Benchmark FlashMLA across batch sizes and sequence lengths (prefill / decode).

Measures vLLM (CUDA), original Triton, and FlashMLA auto latencies,
then prints a formatted table with speedup ratios.

Usage:
  # Decode (--mode decode, default):
  PYTHONPATH=/workspace/FlagGems-vllm/src \
  /workspace/gemms_env/bin/python tools/benchmark_flash_mla.py --warmup 5 --iter 20

  # Peak long-KV sweep:
  PYTHONPATH=/workspace/FlagGems-vllm/src \
  /workspace/gemms_env/bin/python tools/benchmark_flash_mla.py --mode peak --skip-triton --warmup 3 --iter 10

  # Prefill (--mode prefill):
  PYTHONPATH=/workspace/FlagGems-vllm/src \
  /workspace/gemms_env/bin/python tools/benchmark_flash_mla.py --warmup 5 --iter 20 --mode prefill
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from typing import Callable

import torch
import triton

import flaggems_vllm


def _reset_triton_allocator():
    """Reset global Triton allocator set by TLE path, which can interfere with
    subsequent raw Triton kernel launches in batch benchmarks."""
    try:
        triton.set_allocator(None)
    except Exception:
        pass


def _select_flash_mla_variant(variant: str) -> None:
    """Select the FlashMLA implementation under benchmark.

    FlashMLA now defaults to the TLE auto path when no env var is set.
    Use only the variant selector here:
      - auto: benchmark the default auto-selected TLE path.
      - triton: benchmark the original Triton implementation.
    Clear the legacy global TLE switch so a previous measurement does not
    disable the following auto run in the same Python process.
    """
    os.environ.pop("FLAGGEMS_VLLM_FLASH_MLA_TLE", None)
    os.environ["FLAGGEMS_VLLM_FLASH_MLA_TLE_VARIANT"] = variant


def make_inputs(seqlen: int, batch: int, s_q: int, device: str = "cuda"):
    dtype = torch.bfloat16
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


def _unpack_inputs(args_tuple):
    return args_tuple


def make_vllm_run_only(args_tuple):
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
        b,
        s_q,
        cache_seqlens,
        h_q,
        _h_kv,
        d,
        dv,
        causal,
    ) = _unpack_inputs(args_tuple)
    sched_meta, _ = get_mla_metadata()
    out = torch.empty((b, s_q, h_q, dv), dtype=q.dtype, device=q.device)

    def fn():
        flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            sched_meta,
            None,
            softmax_scale=1 / math.sqrt(d),
            causal=causal,
            out=out,
        )
        return out

    fn()
    torch.cuda.synchronize()
    return fn


def make_tle_plan_run_only(args_tuple):
    from flaggems_vllm.ops.flash_mla import get_flash_mla_tle_decode_plan

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
    ) = _unpack_inputs(args_tuple)
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

    def fn():
        return plan.run(
            q=q,
            blocked_k=blocked_k,
            block_table=block_table,
            update_metadata=False,
            out=out,
        )

    fn()
    torch.cuda.synchronize()
    return fn, plan


def make_tle_sched_only(args_tuple):
    from flaggems_vllm.ops.flash_mla import get_flash_mla_tle_decode_plan

    (
        q,
        _block_table,
        _blocked_k,
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
    ) = _unpack_inputs(args_tuple)
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

    def fn():
        plan.plan(cache_seqlens)

    fn()
    torch.cuda.synchronize()
    return fn


def measure(fn: Callable, warmup: int, iters: int):
    """GPU kernel time in ms via triton.testing.do_bench (median)."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=iters, return_mode="median")


def _safe_cleanup():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def format_val(v) -> str:
    if isinstance(v, str):
        return f"{v:>10}"
    return f"{v:10.4f}"


def format_ratio(v) -> str:
    if isinstance(v, str):
        return f"{v:>11}"
    return f"{v:10.3f}x"


def format_tflops(v) -> str:
    if isinstance(v, str):
        return f"{v:>12}"
    return f"{v:11.2f}"


def _human_int(n: int) -> str:
    if n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}M"
    if n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def _nominal_tflops(lat_ms, batch: int, s_kv: int, s_q: int, h_q: int, d_qk: int, dv: int = 512):
    if isinstance(lat_ms, str) or lat_ms <= 0:
        return "NaN"
    flops = 2 * batch * s_q * h_q * s_kv * (d_qk + dv)
    return flops / lat_ms / 1e9


def _parse_peak_shape(value: str) -> tuple[int, int]:
    try:
        batch_s, skv_s = value.split(":", 1)
        return int(batch_s), int(skv_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"expected B:S_KV, for example 16:2097152, got {value!r}"
        ) from exc


def _print_table(header: str, rows: list):
    print(header)
    print("-" * len(header))
    for row in rows:
        print(row)
    print("-" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--mode", choices=("decode", "prefill", "single", "peak"), default="decode")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--s-kv", type=int, default=4096)
    parser.add_argument("--h-q", type=int, default=128)
    parser.add_argument("--d-qk", type=int, default=576)
    parser.add_argument("--peak-shape", action="append", type=_parse_peak_shape, default=None,
                        help=("Long-KV peak shape as B:S_KV. Can be repeated. "
                              "Default for --mode peak: 16:2097152, 32:2097152, 8:8388608"))
    parser.add_argument("--bench-flow", choices=("full", "run-only", "split"), default="full",
                        help=("full: current single-call benchmark; run-only: prime metadata "
                              "outside timing and benchmark only repeated run; split: also "
                              "benchmark TLE scheduler metadata update separately"))
    parser.add_argument("--skip-triton", action="store_true", default=False,
                        help="Skip Triton baseline, only benchmark vLLM and TLE")
    ns = parser.parse_args()

    if ns.mode == "single":
        batch_sizes = [ns.batch]
        s_kv_list = [ns.s_kv]
        s_q = 1
        h_q = ns.h_q
        d_qk = ns.d_qk
        shape_pairs = [(ns.batch, ns.s_kv)]
    elif ns.mode == "decode":
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        s_kv_list = [4096, 8192, 32768]
        s_q = 1
        shape_pairs = [(b, s_kv) for b in batch_sizes for s_kv in s_kv_list]
    elif ns.mode == "peak":
        shape_pairs = ns.peak_shape or [
            (2, 2 * 1024 * 1024),
            (4, 2 * 1024 * 1024),
            (8, 2 * 1024 * 1024),
            (2, 4 * 1024 * 1024),
            (2, 8 * 1024 * 1024),
            (4, 4 * 1024 * 1024),
            (1, 8 * 1024 * 1024),
        ]
        batch_sizes = []
        s_kv_list = []
        s_q = 1
    else:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        s_kv_list = [4096, 8192, 32768]
        s_q = 4096
        shape_pairs = [(b, s_kv) for b in batch_sizes for s_kv in s_kv_list]

    if ns.mode != "single":
        h_q = 128
        d_qk = 576

    all_results = []
    errors = []

    def _failed(v):
        return isinstance(v, str)

    fatal_cuda_error = False

    def _measure_or_error(label: str, batch: int, s_kv: int, fn: Callable):
        nonlocal fatal_cuda_error
        print(f"  BEGIN batch={batch:3d} s_kv={s_kv:5d} {label}", file=sys.stderr, flush=True)
        try:
            return measure(fn, ns.warmup, ns.iter)
        except Exception as e:
            message = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(
                f"  FAIL  batch={batch:3d} s_kv={s_kv:5d} {label}: {type(e).__name__}: {message}",
                file=sys.stderr,
                flush=True,
            )
            if "CUDA error" in str(e) or "illegal memory access" in str(e):
                fatal_cuda_error = True
            return f"ERR:{type(e).__name__}"

    def _safe_del_cleanup(*objs):
        for obj in objs:
            try:
                del obj
            except Exception:
                pass
        try:
            gc.collect()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        _reset_triton_allocator()

    for batch, s_kv in shape_pairs:
            if fatal_cuda_error:
                break
            _safe_del_cleanup()

            try:
                inputs = make_inputs(s_kv, batch, s_q)
            except Exception as e:
                print(f"  batch={batch:3d} s_kv={s_kv:5d} -- {type(e).__name__}, skipping", file=sys.stderr)
                errors.append((batch, s_kv, f"input:{type(e).__name__}"))
                continue

            if ns.bench_flow != "full":
                _reset_triton_allocator()
                fn_vllm_run = make_vllm_run_only(inputs)
                lat_vllm_run = _measure_or_error("vLLM-run", batch, s_kv, fn_vllm_run)
                if _failed(lat_vllm_run):
                    errors.append((batch, s_kv, f"vLLM-run:{lat_vllm_run}"))

                lat_tle_sched = "-"
                if ns.bench_flow == "split":
                    _select_flash_mla_variant("auto")
                    fn_tle_sched = make_tle_sched_only(inputs)
                    lat_tle_sched = _measure_or_error("TLE-sched", batch, s_kv, fn_tle_sched)
                    if _failed(lat_tle_sched):
                        errors.append((batch, s_kv, f"TLE-sched:{lat_tle_sched}"))

                _select_flash_mla_variant("auto")
                fn_tle_run, _tle_plan = make_tle_plan_run_only(inputs)
                lat_tle_run = _measure_or_error("TLE-run", batch, s_kv, fn_tle_run)
                if _failed(lat_tle_run):
                    errors.append((batch, s_kv, f"TLE-run:{lat_tle_run}"))

                tle_sched_plus_run = "-"
                if not _failed(lat_tle_sched) and not _failed(lat_tle_run):
                    if isinstance(lat_tle_sched, str):
                        tle_sched_plus_run = "-"
                    else:
                        tle_sched_plus_run = lat_tle_sched + lat_tle_run

                all_results.append(
                    (
                        batch,
                        s_kv,
                        s_q,
                        h_q,
                        d_qk,
                        lat_vllm_run,
                        lat_tle_sched,
                        lat_tle_run,
                        tle_sched_plus_run,
                    )
                )
                print(
                    f"  batch={batch:3d} s_kv={s_kv:5d} "
                    f"vLLM-run={format_val(lat_vllm_run)} "
                    f"TLE-sched={format_val(lat_tle_sched)} "
                    f"TLE-run={format_val(lat_tle_run)}",
                    file=sys.stderr,
                )
                _safe_del_cleanup(
                    inputs,
                    fn_vllm_run,
                    fn_tle_sched if ns.bench_flow == "split" else None,
                    fn_tle_run,
                )
                continue

            # --- vLLM ---
            _reset_triton_allocator()
            fn_vllm = lambda: run_vllm(inputs)
            lat_vllm = _measure_or_error("vLLM", batch, s_kv, fn_vllm)
            if _failed(lat_vllm):
                errors.append((batch, s_kv, f"vLLM:{lat_vllm}"))
                print(f"  batch={batch:3d} s_kv={s_kv:5d} vLLM={lat_vllm}", file=sys.stderr)

            # --- FlashMLA auto (run before Triton to avoid allocator pollution) ---
            _select_flash_mla_variant("auto")
            fn_tle = lambda: flaggems_vllm.flash_mla(*inputs)
            lat_tle = _measure_or_error("TLE", batch, s_kv, fn_tle)
            if _failed(lat_tle):
                errors.append((batch, s_kv, f"TLE:{lat_tle}"))
                print(f"  batch={batch:3d} s_kv={s_kv:5d} TLE={lat_tle}", file=sys.stderr)

            # --- Original Triton ---
            fn_triton = None
            if ns.skip_triton:
                lat_triton = "-"
            else:
                _reset_triton_allocator()
                _select_flash_mla_variant("triton")
                fn_triton = lambda: flaggems_vllm.flash_mla(*inputs)
                lat_triton = _measure_or_error("Triton", batch, s_kv, fn_triton)
                if _failed(lat_triton):
                    errors.append((batch, s_kv, f"Triton:{lat_triton}"))
                    print(f"  batch={batch:3d} s_kv={s_kv:5d} Triton={lat_triton}", file=sys.stderr)

            _safe_del_cleanup(inputs, fn_vllm, fn_triton, fn_tle)

            # Compute ratios (NaN if any measurement failed)
            if ns.skip_triton:
                triton_vs_vllm = "-"
                tle_vs_triton = "-"
            elif _failed(lat_vllm) or _failed(lat_triton):
                triton_vs_vllm = "NaN"
            else:
                triton_vs_vllm = lat_vllm / lat_triton if lat_triton > 0 else 0

            if _failed(lat_vllm) or _failed(lat_tle):
                tle_vs_vllm = "NaN"
            else:
                tle_vs_vllm = lat_vllm / lat_tle if lat_tle > 0 else 0

            if ns.skip_triton:
                tle_vs_triton = "-"
            elif _failed(lat_triton) or _failed(lat_tle):
                tle_vs_triton = "NaN"
            else:
                tle_vs_triton = lat_triton / lat_tle if lat_tle > 0 else 0

            all_results.append(
                (batch, s_kv, s_q, h_q, d_qk,
                 lat_vllm, lat_triton, lat_tle,
                 triton_vs_vllm, tle_vs_vllm, tle_vs_triton)
            )
            print(f"  batch={batch:3d} s_kv={s_kv:5d} "
                  f"vLLM={format_val(lat_vllm)} Triton={format_val(lat_triton)} TLE={format_val(lat_tle)}",
                  file=sys.stderr)
    if fatal_cuda_error:
        print("  Stopping after CUDA error; CUDA context is no longer reliable.", file=sys.stderr)

    if not all_results:
        print("No successful results.", file=sys.stderr)
        return

    if ns.bench_flow != "full":
        if ns.bench_flow == "split":
            header = (
                f"{'batch':>6}  {'s_kv':>6}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                f"{'vLLM-run(ms)':>13}  {'TLE-sched(ms)':>13}  "
                f"{'TLE-run(ms)':>11}  {'TLE-s+r(ms)':>12}  "
                f"{'TLErun/vLLM':>12}"
            )
        else:
            header = (
                f"{'batch':>6}  {'s_kv':>6}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                f"{'vLLM-run(ms)':>13}  {'TLE-run(ms)':>11}  "
                f"{'TLErun/vLLM':>12}"
            )
        rows = []
        valid = []
        for (
            batch,
            s_kv,
            s_q,
            h_q,
            d_qk,
            lat_vllm_run,
            lat_tle_sched,
            lat_tle_run,
            tle_sched_plus_run,
        ) in all_results:
            tle_vs_vllm = (
                lat_vllm_run / lat_tle_run
                if not _failed(lat_vllm_run)
                and not _failed(lat_tle_run)
                and lat_tle_run > 0
                else "NaN"
            )
            if ns.bench_flow == "split":
                rows.append(
                    f"{batch:6d}  {s_kv:6d}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                    f"{format_val(lat_vllm_run)}  {format_val(lat_tle_sched)}  "
                    f"{format_val(lat_tle_run)}  {format_val(tle_sched_plus_run)}  "
                    f"{format_ratio(tle_vs_vllm)}"
                )
            else:
                rows.append(
                    f"{batch:6d}  {s_kv:6d}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                    f"{format_val(lat_vllm_run)}  {format_val(lat_tle_run)}  "
                    f"{format_ratio(tle_vs_vllm)}"
                )
            if not _failed(lat_vllm_run) and not _failed(lat_tle_run):
                valid.append((lat_vllm_run, lat_tle_run))

        print()
        _print_table(header, rows)
        if valid:
            avg_vllm_run = sum(x for x, _ in valid) / len(valid)
            avg_tle_run = sum(y for _, y in valid) / len(valid)
            avg_ratio = avg_vllm_run / avg_tle_run if avg_tle_run > 0 else 0
            if ns.bench_flow == "split":
                print(
                    f"{'Avg':>6}  {'-':>6}  {'-':>4}  {'-':>4}  {'-':>4}  "
                    f"{avg_vllm_run:13.4f}  {'-':>13}  "
                    f"{avg_tle_run:11.4f}  {'-':>12}  "
                    f"{avg_ratio:11.3f}x"
                )
            else:
                print(
                    f"{'Avg':>6}  {'-':>6}  {'-':>4}  {'-':>4}  {'-':>4}  "
                    f"{avg_vllm_run:13.4f}  {avg_tle_run:11.4f}  "
                    f"{avg_ratio:11.3f}x"
                )
        print(
            "\nNote: vLLM-run is measured after one untimed flash_mla_with_kvcache "
            "call initializes FlashMLASchedMeta. TLE-run is measured after "
            "plan.plan(cache_seqlens). Original Triton is intentionally omitted "
            "from run-only/split modes because it has no sched/run API in this script."
        )
        return

    # --- Print table ---
    if ns.skip_triton and ns.mode == "peak":
        header = (f"{'batch':>6}  {'s_kv':>8}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                  f"{'vLLM(ms)':>10}  {'TLE(ms)':>10}  "
                  f"{'vLLM TFLOPS':>12}  {'TLE TFLOPS':>12}  "
                  f"{'TLE/vLLM':>11}")
    elif ns.skip_triton:
        header = (f"{'batch':>6}  {'s_kv':>6}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                  f"{'vLLM(ms)':>10}  {'TLE(ms)':>10}  "
                  f"{'TLE/vLLM':>11}")
    elif ns.mode == "peak":
        header = (f"{'batch':>6}  {'s_kv':>8}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                  f"{'vLLM(ms)':>10}  {'Triton(ms)':>10}  {'TLE(ms)':>10}  "
                  f"{'vLLM TFLOPS':>12}  {'TLE TFLOPS':>12}  "
                  f"{'Triton/vLLM':>11}  {'TLE/vLLM':>11}  {'TLE/Triton':>11}")
    else:
        header = (f"{'batch':>6}  {'s_kv':>6}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                  f"{'vLLM(ms)':>10}  {'Triton(ms)':>10}  {'TLE(ms)':>10}  "
                  f"{'Triton/vLLM':>11}  {'TLE/vLLM':>11}  {'TLE/Triton':>11}")

    rows = []
    for (batch, s_kv, s_q, h_q, d_qk,
         lat_vllm, lat_triton, lat_tle,
         triton_vs_vllm, tle_vs_vllm, tle_vs_triton) in all_results:
        if ns.skip_triton:
            vals = [format_val(v) for v in [batch, s_kv, s_q, h_q, d_qk, lat_vllm, lat_tle]]
            ratios = [format_ratio(v) for v in [tle_vs_vllm]]
            if ns.mode == "peak":
                vllm_tflops = _nominal_tflops(lat_vllm, batch, s_kv, s_q, h_q, d_qk)
                tle_tflops = _nominal_tflops(lat_tle, batch, s_kv, s_q, h_q, d_qk)
                rows.append(f"{batch:6d}  {_human_int(s_kv):>8}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                            f"{vals[5]}  {vals[6]}  "
                            f"{format_tflops(vllm_tflops)}  {format_tflops(tle_tflops)}  "
                            f"{ratios[0]}")
            else:
                rows.append(f"{batch:6d}  {s_kv:6d}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                            f"{vals[5]}  {vals[6]}  "
                            f"{ratios[0]}")
        else:
            vals = [format_val(v) for v in [batch, s_kv, s_q, h_q, d_qk, lat_vllm, lat_triton, lat_tle]]
            ratios = [format_ratio(v) for v in [triton_vs_vllm, tle_vs_vllm, tle_vs_triton]]
            if ns.mode == "peak":
                vllm_tflops = _nominal_tflops(lat_vllm, batch, s_kv, s_q, h_q, d_qk)
                tle_tflops = _nominal_tflops(lat_tle, batch, s_kv, s_q, h_q, d_qk)
                rows.append(f"{batch:6d}  {_human_int(s_kv):>8}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                            f"{vals[5]}  {vals[6]}  {vals[7]}  "
                            f"{format_tflops(vllm_tflops)}  {format_tflops(tle_tflops)}  "
                            f"{ratios[0]}  {ratios[1]}  {ratios[2]}")
            else:
                rows.append(f"{batch:6d}  {s_kv:6d}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                            f"{vals[5]}  {vals[6]}  {vals[7]}  "
                            f"{ratios[0]}  {ratios[1]}  {ratios[2]}")

    print()
    _print_table(header, rows)

    # --- Avg row (exclude errors) ---
    if ns.skip_triton:
        valid = [r for r in all_results
                 if not isinstance(r[5], str) and not isinstance(r[7], str)]
        if valid:
            avg_vllm = sum(r[5] for r in valid) / len(valid)
            avg_tle = sum(r[7] for r in valid) / len(valid)
            avg_tle_vs_vllm = avg_vllm / avg_tle if avg_tle > 0 else 0
            print(f"{'Avg':>6}  {'-':>6}  {'-':>4}  {'-':>4}  {'-':>4}  "
                  f"{avg_vllm:10.4f}  {avg_tle:10.4f}  "
                  f"{avg_tle_vs_vllm:10.3f}x")
    else:
        valid = [r for r in all_results
                 if not isinstance(r[5], str) and not isinstance(r[6], str) and not isinstance(r[7], str)]
        if valid:
            avg_vllm = sum(r[5] for r in valid) / len(valid)
            avg_triton = sum(r[6] for r in valid) / len(valid)
            avg_tle = sum(r[7] for r in valid) / len(valid)
            avg_triton_vs_vllm = avg_vllm / avg_triton if avg_triton > 0 else 0
            avg_tle_vs_vllm = avg_vllm / avg_tle if avg_tle > 0 else 0
            avg_tle_vs_triton = avg_triton / avg_tle if avg_tle > 0 else 0
            print(f"{'Avg':>6}  {'-':>6}  {'-':>4}  {'-':>4}  {'-':>4}  "
                  f"{avg_vllm:10.4f}  {avg_triton:10.4f}  {avg_tle:10.4f}  "
                  f"{avg_triton_vs_vllm:10.3f}x  {avg_tle_vs_vllm:10.3f}x  {avg_tle_vs_triton:10.3f}x")

    if errors:
        print(f"\n  Failed: {errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
