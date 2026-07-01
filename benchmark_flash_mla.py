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
import importlib.util
import json
import math
import os
import sys
import time
from typing import Callable

import torch
import triton

import flaggems_vllm


_TILELANG_EXAMPLE_PATH = "/workspace/example_decode_paged.py"
_TILELANG_DISPLAY_NAME = "TileLang-Pipeline"
_TILELANG_MS_COL_WIDTH = max(25, len(_TILELANG_DISPLAY_NAME + "(ms)"))
_TILELANG_RUN_COL_WIDTH = max(25, len(_TILELANG_DISPLAY_NAME + "-run(ms)"))
_TILELANG_TFLOPS_COL_WIDTH = max(25, len(_TILELANG_DISPLAY_NAME + " TFLOPS"))
_PLOT_IMPL_LABELS = {
    "VLLM": "DeepSeek Cuda",
    "FLASHINFER": "Flashinfer",
    _TILELANG_DISPLAY_NAME: "Tilelang-Pipeline",
    "TRITON": "Triton",
    "TLE": "Tle",
}
_tilelang_example_module = None


def _reset_triton_allocator():
    """Reset global Triton allocator set by TLE path, which can interfere with
    subsequent raw Triton kernel launches in batch benchmarks."""
    try:
        triton.set_allocator(None)
    except Exception:
        pass


def _load_tilelang_example():
    global _tilelang_example_module
    if _tilelang_example_module is not None:
        return _tilelang_example_module
    if not os.path.exists(_TILELANG_EXAMPLE_PATH):
        raise FileNotFoundError(_TILELANG_EXAMPLE_PATH)
    spec = importlib.util.spec_from_file_location(
        "example_decode_paged", _TILELANG_EXAMPLE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {_TILELANG_EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _tilelang_example_module = module
    return module


def _select_flash_mla_variant(variant: str) -> None:
    """Select the FlashMLA implementation under benchmark.

    FlashMLA now defaults to the TLE auto path when no env var is set.
    Use only the variant selector here:
      - auto: benchmark the default auto-selected TLE path.
      - triton: benchmark the original Triton implementation.
    Clear the legacy global TLE switch so a previous measurement does not
    disable the following auto run in the same Python process.
    """
    os.environ["FLAGGEMS_VLLM_FLASH_MLA_TLE_VARIANT"] = variant
    if variant == "triton":
        os.environ["FLAGGEMS_VLLM_FLASH_MLA_TLE"] = "0"
    else:
        os.environ.pop("FLAGGEMS_VLLM_FLASH_MLA_TLE", None)


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


def _flashinfer_mla_args(args_tuple):
    (
        q,
        block_table,
        blocked_k,
        _max_seqlen_pad,
        block_size,
        batch,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    ) = args_tuple
    if s_q != 1:
        raise ValueError("FlashInfer MLA benchmark path only supports decode s_q=1")
    if h_kv != 1:
        raise ValueError("FlashInfer MLA benchmark path expects h_kv=1")
    if d != 576 or dv != 512:
        raise ValueError("FlashInfer MLA benchmark path expects d=576 and dv=512")

    num_pages = torch.div(
        cache_seqlens + block_size - 1, block_size, rounding_mode="floor"
    ).to(torch.int32)
    kv_indptr = torch.empty((batch + 1,), dtype=torch.int32, device=q.device)
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(num_pages, dim=0)
    indices = torch.cat(
        [block_table[i, : int(num_pages[i].item())] for i in range(batch)]
    ).contiguous()
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=q.device)

    q_nope = q[:, 0, :, :dv]
    q_pe = q[:, 0, :, dv:d]
    ckv_cache = blocked_k[:, :, 0, :dv]
    kpe_cache = blocked_k[:, :, 0, dv:d]
    return (
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        qo_indptr,
        kv_indptr,
        indices,
        cache_seqlens,
        block_size,
        batch,
        h_q,
        d,
        dv,
        causal,
    )


def _make_flashinfer_planned(args_tuple):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    (
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        qo_indptr,
        kv_indptr,
        indices,
        cache_seqlens,
        block_size,
        batch,
        h_q,
        d,
        dv,
        causal,
    ) = _flashinfer_mla_args(args_tuple)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q_nope.device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="auto")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        indices,
        cache_seqlens,
        h_q,
        dv,
        d - dv,
        block_size,
        causal,
        1 / math.sqrt(d),
        q_nope.dtype,
        ckv_cache.dtype,
    )
    out = torch.empty((batch, h_q, dv), dtype=q_nope.dtype, device=q_nope.device)

    def fn():
        return wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, out=out)

    return fn


def make_flashinfer_run_only(args_tuple):
    fn = _make_flashinfer_planned(args_tuple)
    fn()
    torch.cuda.synchronize()
    return fn


def run_flashinfer(args_tuple):
    return _make_flashinfer_planned(args_tuple)()


def _tilelang_mla_args(args_tuple):
    (
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
    ) = args_tuple
    if s_q != 1:
        raise ValueError("TileLang MLA benchmark path only supports decode s_q=1")
    if h_kv != 1:
        raise ValueError("TileLang MLA benchmark path expects h_kv=1")
    if d <= dv:
        raise ValueError("TileLang MLA benchmark path expects d > dv")
    if block_size < 64 or block_size % 64 != 0:
        raise ValueError("TileLang MLA benchmark path expects block_size to be a multiple of 64")

    # /workspace/example_decode_paged.py declares TileLang tensor dtype as fp16.
    # Cast once outside the timed path so the measured number is kernel-only.
    q_nope = q[:, 0, :, :dv].contiguous().to(torch.float16)
    q_pe = q[:, 0, :, dv:d].contiguous().to(torch.float16)
    blocked_k_nope = blocked_k[..., :dv].contiguous().to(torch.float16)
    blocked_k_pe = blocked_k[..., dv:d].contiguous().to(torch.float16)
    return (
        q_nope,
        q_pe,
        blocked_k_nope.view(-1, h_kv, dv),
        blocked_k_pe.view(-1, h_kv, d - dv),
        block_table,
        cache_seqlens,
        max_seqlen_pad,
        block_size,
        batch,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    )


def _make_tilelang_planned(args_tuple, num_kv_splits: int = 1):
    module = _load_tilelang_example()
    (
        q_nope,
        q_pe,
        kv_nope,
        k_pe,
        block_table,
        cache_seqlens,
        max_seqlen_pad,
        block_size,
        batch,
        h_q,
        h_kv,
        d,
        dv,
        _causal,
    ) = _tilelang_mla_args(args_tuple)
    dpe = d - dv
    if num_kv_splits < 1:
        raise ValueError("TileLang num_split must be >= 1")
    block_n = 64
    block_h = min(64, h_q // h_kv)
    softmax_scale = d**-0.5
    glse = torch.empty(
        batch, h_q, num_kv_splits, dtype=torch.float16, device=q_nope.device
    )
    out_partial = torch.empty(
        batch, h_q, num_kv_splits, dv, dtype=torch.float16, device=q_nope.device
    )
    kernel = module.mla_decode_tilelang(
        batch,
        h_q,
        h_kv,
        max_seqlen_pad,
        dv,
        dpe,
        block_n,
        block_h,
        num_kv_splits,
        block_size,
        softmax_scale,
    )
    profiler = kernel.get_profiler(tensor_supply_type=module.tilelang.TensorSupplyType.Randn)

    def fn():
        return profiler.func(
            q_nope,
            q_pe,
            kv_nope,
            k_pe,
            block_table,
            cache_seqlens,
            glse,
            out_partial,
        )

    return fn


def make_tilelang_run_only(args_tuple, num_kv_splits: int = 1):
    fn = _make_tilelang_planned(args_tuple, num_kv_splits)
    fn()
    torch.cuda.synchronize()
    return fn


def run_tilelang(args_tuple, num_kv_splits: int = 1):
    return _make_tilelang_planned(args_tuple, num_kv_splits)()


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


def format_val(v, width: int = 10) -> str:
    if isinstance(v, str):
        return f"{v:>{width}}"
    return f"{v:{width}.4f}"


def format_ratio(v) -> str:
    if isinstance(v, str):
        return f"{v:>11}"
    return f"{v:10.3f}x"


def format_tflops(v, width: int = 12) -> str:
    if isinstance(v, str):
        return f"{v:>{width}}"
    return f"{v:{width}.2f}"


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


def _avg_full_tflops(rows: list, latency_idx: int):
    vals = []
    for r in rows:
        tflops = _nominal_tflops(r[latency_idx], r[0], r[1], r[2], r[3], r[4])
        if not isinstance(tflops, str):
            vals.append(tflops)
    return sum(vals) / len(vals) if vals else "NaN"


def _json_number(v):
    if isinstance(v, str):
        return None
    try:
        if math.isnan(float(v)):
            return None
    except Exception:
        return None
    return float(v)


def _full_summary_records(all_results: list, include_triton: bool) -> list[dict]:
    records = []
    impls = [
        ("VLLM", 5),
        ("FLASHINFER", 6),
        (_TILELANG_DISPLAY_NAME, 7),
        ("TRITON", 8),
        ("TLE", 9),
    ]
    if not include_triton:
        impls = [item for item in impls if item[0] != "TRITON"]

    for row in all_results:
        batch, s_kv, s_q, h_q, d_qk = row[:5]
        for impl, idx in impls:
            latency_ms = row[idx]
            tflops = _nominal_tflops(latency_ms, batch, s_kv, s_q, h_q, d_qk)
            records.append(
                {
                    "batch": batch,
                    "s_kv": s_kv,
                    "s_q": s_q,
                    "h_q": h_q,
                    "d_qk": d_qk,
                    "dv": 512,
                    "implementation": impl,
                    "latency_ms": _json_number(latency_ms),
                    "tflops": _json_number(tflops),
                    "status": "ok" if not isinstance(latency_ms, str) else latency_ms,
                }
            )
    return records


def _write_summary_json(path: str, payload: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _plot_tflops_by_batch(records: list[dict], plot_dir: str) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  Plot skipped: matplotlib import failed: {exc}", file=sys.stderr)
        return []

    os.makedirs(plot_dir, exist_ok=True)
    batches = sorted({r["batch"] for r in records})
    impls = ["VLLM", "FLASHINFER", _TILELANG_DISPLAY_NAME, "TRITON", "TLE"]
    written = []
    for batch in batches:
        batch_records = [
            r
            for r in records
            if r["batch"] == batch and r["status"] == "ok" and r["tflops"] is not None
        ]
        if not batch_records:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        measured_s_kv = sorted({r["s_kv"] for r in batch_records})
        x_positions = {s_kv: idx for idx, s_kv in enumerate(measured_s_kv)}
        for impl in impls:
            points = sorted(
                [(r["s_kv"], r["tflops"]) for r in batch_records if r["implementation"] == impl]
            )
            if not points:
                continue
            xs = [x_positions[s_kv] for s_kv, _ in points]
            ys = [tflops for _, tflops in points]
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=_PLOT_IMPL_LABELS.get(impl, impl.capitalize()))
        ax.set_xticks(list(range(len(measured_s_kv))))
        ax.set_xticklabels([str(x) for x in measured_s_kv], rotation=30, ha="right")
        y_ticks = list(ax.get_yticks())
        if 140 not in y_ticks:
            y_ticks.append(140)
            ax.set_yticks(sorted(y_ticks))
        ax.set_title(f"FlashMLA full TFLOPS vs KV length, batch={batch}")
        ax.set_xlabel("KV length")
        ax.set_ylabel("Nominal TFLOPS")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(plot_dir, f"batch_{batch}_tflops.png")
        fig.savefig(out_path)
        plt.close(fig)
        written.append(out_path)
    return written


def _write_full_summary(
    ns,
    all_results: list,
    errors: list,
    include_triton: bool,
) -> None:
    records = _full_summary_records(all_results, include_triton=include_triton)
    plot_files = []
    if not ns.no_plots and ns.mode != "peak":
        plot_files = _plot_tflops_by_batch(records, ns.plot_dir)
    payload = {
        "benchmark": "flash_mla_full",
        "bench_flow": ns.bench_flow,
        "mode": ns.mode,
        "warmup": ns.warmup,
        "iterations": ns.iter,
        "tilelang_name": _TILELANG_DISPLAY_NAME,
        "tilelang_num_split": ns.tilelang_num_split,
        "tflops_definition": "2 * batch * s_q * h_q * s_kv * (d_qk + dv) / latency_ms / 1e9",
        "records": records,
        "plots": plot_files,
        "errors": [
            {"batch": b, "s_kv": s_kv, "error": error}
            for b, s_kv, error in errors
        ],
    }
    _write_summary_json(ns.summary_json, payload)
    print(f"\nSummary JSON: {ns.summary_json}")
    if plot_files:
        print("TFLOPS plots:")
        for path in plot_files:
            print(f"  {path}")


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
    parser.add_argument("--tle-variant", choices=("auto", "qtail_global"), default="auto",
                        help="TLE run variant to benchmark")
    parser.add_argument("--skip-triton", action="store_true", default=False,
                        help="Skip Triton baseline, only benchmark vLLM and TLE")
    parser.add_argument("--skip-flashinfer", action="store_true", default=False,
                        help="Skip FlashInfer MLA baseline")
    parser.add_argument("--skip-tilelang", action="store_true", default=False,
                        help="Skip TileLang MLA baseline from /workspace/example_decode_paged.py")
    parser.add_argument("--tilelang-num-split", type=int, default=1,
                        help="TileLang KV split count for /workspace/example_decode_paged.py")
    parser.add_argument("--summary-json", default="summary.json",
                        help="Write full-mode benchmark records and TFLOPS to this JSON file")
    parser.add_argument("--plot-dir", default="plots",
                        help="Directory for per-batch TFLOPS-vs-KV plots")
    parser.add_argument("--no-plots", action="store_true", default=False,
                        help="Do not generate per-batch TFLOPS plots")
    ns = parser.parse_args()

    if ns.mode == "single":
        batch_sizes = [ns.batch]
        s_kv_list = [ns.s_kv]
        s_q = 1
        h_q = ns.h_q
        d_qk = ns.d_qk
        shape_pairs = [(ns.batch, ns.s_kv)]
    elif ns.mode == "decode":
        batch_sizes = [32, 64, 128, 256, 512]
        s_kv_list = [1024, 2048, 4096, 8192, 16384, 32768]
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
            (2, 8 * 1024 * 1024),
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

                if ns.skip_flashinfer:
                    lat_flashinfer_run = "-"
                    fn_flashinfer_run = None
                else:
                    fn_flashinfer_run = make_flashinfer_run_only(inputs)
                    lat_flashinfer_run = _measure_or_error("FlashInfer-run", batch, s_kv, fn_flashinfer_run)
                    if _failed(lat_flashinfer_run):
                        errors.append((batch, s_kv, f"FlashInfer-run:{lat_flashinfer_run}"))

                if ns.skip_tilelang:
                    lat_tilelang_run = "-"
                    fn_tilelang_run = None
                else:
                    fn_tilelang_run = make_tilelang_run_only(inputs, ns.tilelang_num_split)
                    lat_tilelang_run = _measure_or_error(f"{_TILELANG_DISPLAY_NAME}-run", batch, s_kv, fn_tilelang_run)
                    if _failed(lat_tilelang_run):
                        errors.append((batch, s_kv, f"{_TILELANG_DISPLAY_NAME}-run:{lat_tilelang_run}"))

                lat_tle_sched = "-"
                if ns.bench_flow == "split":
                    _select_flash_mla_variant(ns.tle_variant)
                    fn_tle_sched = make_tle_sched_only(inputs)
                    lat_tle_sched = _measure_or_error("TLE-sched", batch, s_kv, fn_tle_sched)
                    if _failed(lat_tle_sched):
                        errors.append((batch, s_kv, f"TLE-sched:{lat_tle_sched}"))

                _select_flash_mla_variant(ns.tle_variant)
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
                        lat_flashinfer_run,
                        lat_tilelang_run,
                        lat_tle_sched,
                        lat_tle_run,
                        tle_sched_plus_run,
                    )
                )
                print(
                    f"  batch={batch:3d} s_kv={s_kv:5d} "
                    f"vLLM-run={format_val(lat_vllm_run)} "
                    f"FlashInfer-run={format_val(lat_flashinfer_run)} "
                    f"{_TILELANG_DISPLAY_NAME}-run={format_val(lat_tilelang_run)} "
                    f"TLE-sched={format_val(lat_tle_sched)} "
                    f"TLE-run={format_val(lat_tle_run)}",
                    file=sys.stderr,
                )
                _safe_del_cleanup(
                    inputs,
                    fn_vllm_run,
                    fn_flashinfer_run,
                    fn_tilelang_run,
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

            # --- FlashInfer ---
            if ns.skip_flashinfer:
                lat_flashinfer = "-"
            else:
                fn_flashinfer = lambda: run_flashinfer(inputs)
                lat_flashinfer = _measure_or_error("FlashInfer", batch, s_kv, fn_flashinfer)
                if _failed(lat_flashinfer):
                    errors.append((batch, s_kv, f"FlashInfer:{lat_flashinfer}"))
                    print(f"  batch={batch:3d} s_kv={s_kv:5d} FlashInfer={lat_flashinfer}", file=sys.stderr)

            # --- TileLang ---
            if ns.skip_tilelang:
                lat_tilelang = "-"
            else:
                fn_tilelang = lambda: run_tilelang(inputs, ns.tilelang_num_split)
                lat_tilelang = _measure_or_error(_TILELANG_DISPLAY_NAME, batch, s_kv, fn_tilelang)
                if _failed(lat_tilelang):
                    errors.append((batch, s_kv, f"{_TILELANG_DISPLAY_NAME}:{lat_tilelang}"))
                    print(f"  batch={batch:3d} s_kv={s_kv:5d} {_TILELANG_DISPLAY_NAME}={lat_tilelang}", file=sys.stderr)

            # --- FlashMLA auto (run before Triton to avoid allocator pollution) ---
            _select_flash_mla_variant(ns.tle_variant)
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

            _safe_del_cleanup(
                inputs,
                fn_vllm,
                fn_flashinfer if not ns.skip_flashinfer else None,
                fn_tilelang if not ns.skip_tilelang else None,
                fn_triton,
                fn_tle,
            )

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

            if _failed(lat_vllm) or _failed(lat_flashinfer):
                flashinfer_vs_vllm = "NaN"
            else:
                flashinfer_vs_vllm = lat_vllm / lat_flashinfer if lat_flashinfer > 0 else 0

            if _failed(lat_vllm) or _failed(lat_tilelang):
                tilelang_vs_vllm = "NaN"
            else:
                tilelang_vs_vllm = lat_vllm / lat_tilelang if lat_tilelang > 0 else 0

            if ns.skip_triton:
                tle_vs_triton = "-"
            elif _failed(lat_triton) or _failed(lat_tle):
                tle_vs_triton = "NaN"
            else:
                tle_vs_triton = lat_triton / lat_tle if lat_tle > 0 else 0

            all_results.append(
                (batch, s_kv, s_q, h_q, d_qk,
                 lat_vllm, lat_flashinfer, lat_tilelang, lat_triton, lat_tle,
                 flashinfer_vs_vllm, tilelang_vs_vllm, triton_vs_vllm, tle_vs_vllm, tle_vs_triton)
            )
            print(f"  batch={batch:3d} s_kv={s_kv:5d} "
                  f"vLLM={format_val(lat_vllm)} FlashInfer={format_val(lat_flashinfer)} "
                  f"{_TILELANG_DISPLAY_NAME}={format_val(lat_tilelang)} "
                  f"Triton={format_val(lat_triton)} TLE={format_val(lat_tle)}",
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
                f"{'vLLM-run(ms)':>13}  {'FI-run(ms)':>10}  {_TILELANG_DISPLAY_NAME + '-run(ms)':>{_TILELANG_RUN_COL_WIDTH}}  {'TLE-sched(ms)':>13}  "
                f"{'TLE-run(ms)':>11}  {'TLE-s+r(ms)':>12}  "
                f"{'TLE/vLLM':>9}  {'TLE/FI':>9}  {'TLE/TL':>9}"
            )
        else:
            header = (
                f"{'batch':>6}  {'s_kv':>6}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                f"{'vLLM-run(ms)':>13}  {'FI-run(ms)':>10}  {_TILELANG_DISPLAY_NAME + '-run(ms)':>{_TILELANG_RUN_COL_WIDTH}}  {'TLE-run(ms)':>11}  "
                f"{'TLE/vLLM':>9}  {'TLE/FI':>9}  {'TLE/TL':>9}"
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
            lat_flashinfer_run,
            lat_tilelang_run,
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
            tle_vs_flashinfer = (
                lat_flashinfer_run / lat_tle_run
                if not _failed(lat_flashinfer_run)
                and not _failed(lat_tle_run)
                and lat_tle_run > 0
                else "NaN"
            )
            tle_vs_tilelang = (
                lat_tilelang_run / lat_tle_run
                if not _failed(lat_tilelang_run)
                and not _failed(lat_tle_run)
                and lat_tle_run > 0
                else "NaN"
            )
            if ns.bench_flow == "split":
                rows.append(
                    f"{batch:6d}  {s_kv:6d}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                    f"{format_val(lat_vllm_run)}  {format_val(lat_flashinfer_run)}  "
                    f"{format_val(lat_tilelang_run, _TILELANG_RUN_COL_WIDTH)}  "
                    f"{format_val(lat_tle_sched)}  "
                    f"{format_val(lat_tle_run)}  {format_val(tle_sched_plus_run)}  "
                    f"{format_ratio(tle_vs_vllm)}  {format_ratio(tle_vs_flashinfer)}  {format_ratio(tle_vs_tilelang)}"
                )
            else:
                rows.append(
                    f"{batch:6d}  {s_kv:6d}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                    f"{format_val(lat_vllm_run)}  {format_val(lat_flashinfer_run)}  "
                    f"{format_val(lat_tilelang_run, _TILELANG_RUN_COL_WIDTH)}  "
                    f"{format_val(lat_tle_run)}  "
                    f"{format_ratio(tle_vs_vllm)}  {format_ratio(tle_vs_flashinfer)}  {format_ratio(tle_vs_tilelang)}"
                )
            if not _failed(lat_vllm_run) and not _failed(lat_tle_run):
                valid.append((lat_vllm_run, lat_flashinfer_run, lat_tilelang_run, lat_tle_run))

        print()
        _print_table(header, rows)
        if valid:
            avg_vllm_run = sum(x for x, _, _, _ in valid) / len(valid)
            valid_fi = [fi for _, fi, _, _ in valid if not _failed(fi)]
            avg_flashinfer_run = (
                sum(valid_fi) / len(valid_fi) if valid_fi else "NaN"
            )
            valid_tl = [tl for _, _, tl, _ in valid if not _failed(tl)]
            avg_tilelang_run = (
                sum(valid_tl) / len(valid_tl) if valid_tl else "NaN"
            )
            avg_tle_run = sum(y for _, _, _, y in valid) / len(valid)
            avg_ratio = avg_vllm_run / avg_tle_run if avg_tle_run > 0 else 0
            avg_fi_ratio = (
                avg_flashinfer_run / avg_tle_run
                if not _failed(avg_flashinfer_run) and avg_tle_run > 0
                else "NaN"
            )
            avg_tl_ratio = (
                avg_tilelang_run / avg_tle_run
                if not _failed(avg_tilelang_run) and avg_tle_run > 0
                else "NaN"
            )
            if ns.bench_flow == "split":
                print(
                    f"{'Avg':>6}  {'-':>6}  {'-':>4}  {'-':>4}  {'-':>4}  "
                    f"{avg_vllm_run:13.4f}  {format_val(avg_flashinfer_run)}  {format_val(avg_tilelang_run, _TILELANG_RUN_COL_WIDTH)}  {'-':>13}  "
                    f"{avg_tle_run:11.4f}  {'-':>12}  "
                    f"{format_ratio(avg_ratio)}  {format_ratio(avg_fi_ratio)}  {format_ratio(avg_tl_ratio)}"
                )
            else:
                print(
                    f"{'Avg':>6}  {'-':>6}  {'-':>4}  {'-':>4}  {'-':>4}  "
                    f"{avg_vllm_run:13.4f}  {format_val(avg_flashinfer_run)}  {format_val(avg_tilelang_run, _TILELANG_RUN_COL_WIDTH)}  {avg_tle_run:11.4f}  "
                    f"{format_ratio(avg_ratio)}  {format_ratio(avg_fi_ratio)}  {format_ratio(avg_tl_ratio)}"
                )
        print(
            "\nNote: vLLM-run is measured after one untimed flash_mla_with_kvcache "
            "call initializes FlashMLASchedMeta. FlashInfer-run is measured after "
            f"BatchMLAPagedAttentionWrapper.plan. {_TILELANG_DISPLAY_NAME}-run is measured after "
            "TileLang JIT compile, fp16 input casting, buffer allocation, and one "
            "untimed kernel call. TLE-run is measured after "
            "plan.plan(cache_seqlens). Original Triton is intentionally omitted "
            "from run-only/split modes because it has no sched/run API in this script."
        )
        return

    # --- Print table ---
    if ns.skip_triton and ns.mode == "peak":
        header = (f"{'batch':>6}  {'s_kv':>8}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                  f"{'vLLM(ms)':>10}  {'FI(ms)':>10}  {_TILELANG_DISPLAY_NAME + '(ms)':>{_TILELANG_MS_COL_WIDTH}}  {'TLE(ms)':>10}  "
                  f"{'vLLM TFLOPS':>12}  {'FI TFLOPS':>12}  {_TILELANG_DISPLAY_NAME + ' TFLOPS':>{_TILELANG_TFLOPS_COL_WIDTH}}  {'TLE TFLOPS':>12}  "
                  f"{'FI/vLLM':>11}  {'TL/vLLM':>11}  {'TLE/vLLM':>11}  {'TLE/FI':>11}  {'TLE/TL':>11}")
    elif ns.skip_triton:
        header = (f"{'batch':>6}  {'s_kv':>6}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                  f"{'vLLM(ms)':>10}  {'FI(ms)':>10}  {_TILELANG_DISPLAY_NAME + '(ms)':>{_TILELANG_MS_COL_WIDTH}}  {'TLE(ms)':>10}  "
                  f"{'vLLM TFLOPS':>12}  {'FI TFLOPS':>12}  {_TILELANG_DISPLAY_NAME + ' TFLOPS':>{_TILELANG_TFLOPS_COL_WIDTH}}  {'TLE TFLOPS':>12}  "
                  f"{'FI/vLLM':>11}  {'TL/vLLM':>11}  {'TLE/vLLM':>11}  {'TLE/FI':>11}  {'TLE/TL':>11}")
    elif ns.mode == "peak":
        header = (f"{'batch':>6}  {'s_kv':>8}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                  f"{'vLLM(ms)':>10}  {'FI(ms)':>10}  {_TILELANG_DISPLAY_NAME + '(ms)':>{_TILELANG_MS_COL_WIDTH}}  {'Triton(ms)':>10}  {'TLE(ms)':>10}  "
                  f"{'vLLM TFLOPS':>12}  {'FI TFLOPS':>12}  {_TILELANG_DISPLAY_NAME + ' TFLOPS':>{_TILELANG_TFLOPS_COL_WIDTH}}  {'TLE TFLOPS':>12}  "
                  f"{'FI/vLLM':>11}  {'TL/vLLM':>11}  {'Triton/vLLM':>11}  {'TLE/vLLM':>11}  {'TLE/FI':>11}  {'TLE/TL':>11}  {'TLE/Triton':>11}")
    else:
        header = (f"{'batch':>6}  {'s_kv':>6}  {'s_q':>4}  {'h_q':>4}  {'d_qk':>4}  "
                  f"{'vLLM(ms)':>10}  {'FI(ms)':>10}  {_TILELANG_DISPLAY_NAME + '(ms)':>{_TILELANG_MS_COL_WIDTH}}  {'Triton(ms)':>10}  {'TLE(ms)':>10}  "
                  f"{'vLLM TFLOPS':>12}  {'FI TFLOPS':>12}  {_TILELANG_DISPLAY_NAME + ' TFLOPS':>{_TILELANG_TFLOPS_COL_WIDTH}}  {'TLE TFLOPS':>12}  "
                  f"{'FI/vLLM':>11}  {'TL/vLLM':>11}  {'Triton/vLLM':>11}  {'TLE/vLLM':>11}  {'TLE/FI':>11}  {'TLE/TL':>11}  {'TLE/Triton':>11}")

    rows = []
    for (batch, s_kv, s_q, h_q, d_qk,
         lat_vllm, lat_flashinfer, lat_tilelang, lat_triton, lat_tle,
         flashinfer_vs_vllm, tilelang_vs_vllm, triton_vs_vllm,
         tle_vs_vllm, tle_vs_triton) in all_results:
        tle_vs_flashinfer = (
            lat_flashinfer / lat_tle
            if not _failed(lat_flashinfer)
            and not _failed(lat_tle)
            and lat_tle > 0
            else "NaN"
        )
        tle_vs_tilelang = (
            lat_tilelang / lat_tle
            if not _failed(lat_tilelang)
            and not _failed(lat_tle)
            and lat_tle > 0
            else "NaN"
        )
        vllm_tflops = _nominal_tflops(lat_vllm, batch, s_kv, s_q, h_q, d_qk)
        flashinfer_tflops = _nominal_tflops(lat_flashinfer, batch, s_kv, s_q, h_q, d_qk)
        tilelang_tflops = _nominal_tflops(lat_tilelang, batch, s_kv, s_q, h_q, d_qk)
        tle_tflops = _nominal_tflops(lat_tle, batch, s_kv, s_q, h_q, d_qk)
        if ns.skip_triton:
            vals = [format_val(v) for v in [batch, s_kv, s_q, h_q, d_qk, lat_vllm, lat_flashinfer, lat_tilelang, lat_tle]]
            ratios = [format_ratio(v) for v in [flashinfer_vs_vllm, tilelang_vs_vllm, tle_vs_vllm, tle_vs_flashinfer, tle_vs_tilelang]]
            if ns.mode == "peak":
                rows.append(f"{batch:6d}  {_human_int(s_kv):>8}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                            f"{vals[5]}  {vals[6]}  {format_val(lat_tilelang, _TILELANG_MS_COL_WIDTH)}  {vals[8]}  "
                            f"{format_tflops(vllm_tflops)}  {format_tflops(flashinfer_tflops)}  {format_tflops(tilelang_tflops, _TILELANG_TFLOPS_COL_WIDTH)}  {format_tflops(tle_tflops)}  "
                            f"{ratios[0]}  {ratios[1]}  {ratios[2]}  {ratios[3]}  {ratios[4]}")
            else:
                rows.append(f"{batch:6d}  {s_kv:6d}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                            f"{vals[5]}  {vals[6]}  {format_val(lat_tilelang, _TILELANG_MS_COL_WIDTH)}  {vals[8]}  "
                            f"{format_tflops(vllm_tflops)}  {format_tflops(flashinfer_tflops)}  {format_tflops(tilelang_tflops, _TILELANG_TFLOPS_COL_WIDTH)}  {format_tflops(tle_tflops)}  "
                            f"{ratios[0]}  {ratios[1]}  {ratios[2]}  {ratios[3]}  {ratios[4]}")
        else:
            vals = [format_val(v) for v in [batch, s_kv, s_q, h_q, d_qk, lat_vllm, lat_flashinfer, lat_tilelang, lat_triton, lat_tle]]
            ratios = [format_ratio(v) for v in [flashinfer_vs_vllm, tilelang_vs_vllm, triton_vs_vllm, tle_vs_vllm, tle_vs_flashinfer, tle_vs_tilelang, tle_vs_triton]]
            if ns.mode == "peak":
                rows.append(f"{batch:6d}  {_human_int(s_kv):>8}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                            f"{vals[5]}  {vals[6]}  {format_val(lat_tilelang, _TILELANG_MS_COL_WIDTH)}  {vals[8]}  {vals[9]}  "
                            f"{format_tflops(vllm_tflops)}  {format_tflops(flashinfer_tflops)}  {format_tflops(tilelang_tflops, _TILELANG_TFLOPS_COL_WIDTH)}  {format_tflops(tle_tflops)}  "
                            f"{ratios[0]}  {ratios[1]}  {ratios[2]}  {ratios[3]}  {ratios[4]}  {ratios[5]}  {ratios[6]}")
            else:
                rows.append(f"{batch:6d}  {s_kv:6d}  {s_q:4d}  {h_q:4d}  {d_qk:4d}  "
                            f"{vals[5]}  {vals[6]}  {format_val(lat_tilelang, _TILELANG_MS_COL_WIDTH)}  {vals[8]}  {vals[9]}  "
                            f"{format_tflops(vllm_tflops)}  {format_tflops(flashinfer_tflops)}  {format_tflops(tilelang_tflops, _TILELANG_TFLOPS_COL_WIDTH)}  {format_tflops(tle_tflops)}  "
                            f"{ratios[0]}  {ratios[1]}  {ratios[2]}  {ratios[3]}  {ratios[4]}  {ratios[5]}  {ratios[6]}")

    print()
    _print_table(header, rows)

    # --- Avg row (exclude errors) ---
    if ns.skip_triton:
        valid = [r for r in all_results
                 if not isinstance(r[5], str) and not isinstance(r[9], str)]
        if valid:
            avg_vllm = sum(r[5] for r in valid) / len(valid)
            valid_fi = [r[6] for r in valid if not isinstance(r[6], str)]
            avg_flashinfer = sum(valid_fi) / len(valid_fi) if valid_fi else "NaN"
            valid_tl = [r[7] for r in valid if not isinstance(r[7], str)]
            avg_tilelang = sum(valid_tl) / len(valid_tl) if valid_tl else "NaN"
            avg_tle = sum(r[9] for r in valid) / len(valid)
            avg_vllm_tflops = _avg_full_tflops(valid, 5)
            avg_flashinfer_tflops = _avg_full_tflops(valid, 6)
            avg_tilelang_tflops = _avg_full_tflops(valid, 7)
            avg_tle_tflops = _avg_full_tflops(valid, 9)
            avg_flashinfer_vs_vllm = (
                avg_vllm / avg_flashinfer
                if not isinstance(avg_flashinfer, str) and avg_flashinfer > 0
                else "NaN"
            )
            avg_tilelang_vs_vllm = (
                avg_vllm / avg_tilelang
                if not isinstance(avg_tilelang, str) and avg_tilelang > 0
                else "NaN"
            )
            avg_tle_vs_vllm = avg_vllm / avg_tle if avg_tle > 0 else 0
            avg_tle_vs_flashinfer = (
                avg_flashinfer / avg_tle
                if not isinstance(avg_flashinfer, str) and avg_tle > 0
                else "NaN"
            )
            avg_tle_vs_tilelang = (
                avg_tilelang / avg_tle
                if not isinstance(avg_tilelang, str) and avg_tle > 0
                else "NaN"
            )
            print(f"{'Avg':>6}  {'-':>6}  {'-':>4}  {'-':>4}  {'-':>4}  "
                  f"{avg_vllm:10.4f}  {format_val(avg_flashinfer)}  {format_val(avg_tilelang, _TILELANG_MS_COL_WIDTH)}  {avg_tle:10.4f}  "
                  f"{format_tflops(avg_vllm_tflops)}  {format_tflops(avg_flashinfer_tflops)}  {format_tflops(avg_tilelang_tflops, _TILELANG_TFLOPS_COL_WIDTH)}  {format_tflops(avg_tle_tflops)}  "
                  f"{format_ratio(avg_flashinfer_vs_vllm)}  {format_ratio(avg_tilelang_vs_vllm)}  "
                  f"{format_ratio(avg_tle_vs_vllm)}  {format_ratio(avg_tle_vs_flashinfer)}  {format_ratio(avg_tle_vs_tilelang)}")
    else:
        valid = [r for r in all_results
                 if not isinstance(r[5], str) and not isinstance(r[8], str) and not isinstance(r[9], str)]
        if valid:
            avg_vllm = sum(r[5] for r in valid) / len(valid)
            valid_fi = [r[6] for r in valid if not isinstance(r[6], str)]
            avg_flashinfer = sum(valid_fi) / len(valid_fi) if valid_fi else "NaN"
            valid_tl = [r[7] for r in valid if not isinstance(r[7], str)]
            avg_tilelang = sum(valid_tl) / len(valid_tl) if valid_tl else "NaN"
            avg_triton = sum(r[8] for r in valid) / len(valid)
            avg_tle = sum(r[9] for r in valid) / len(valid)
            avg_vllm_tflops = _avg_full_tflops(valid, 5)
            avg_flashinfer_tflops = _avg_full_tflops(valid, 6)
            avg_tilelang_tflops = _avg_full_tflops(valid, 7)
            avg_tle_tflops = _avg_full_tflops(valid, 9)
            avg_flashinfer_vs_vllm = (
                avg_vllm / avg_flashinfer
                if not isinstance(avg_flashinfer, str) and avg_flashinfer > 0
                else "NaN"
            )
            avg_tilelang_vs_vllm = (
                avg_vllm / avg_tilelang
                if not isinstance(avg_tilelang, str) and avg_tilelang > 0
                else "NaN"
            )
            avg_triton_vs_vllm = avg_vllm / avg_triton if avg_triton > 0 else 0
            avg_tle_vs_vllm = avg_vllm / avg_tle if avg_tle > 0 else 0
            avg_tle_vs_flashinfer = (
                avg_flashinfer / avg_tle
                if not isinstance(avg_flashinfer, str) and avg_tle > 0
                else "NaN"
            )
            avg_tle_vs_tilelang = (
                avg_tilelang / avg_tle
                if not isinstance(avg_tilelang, str) and avg_tle > 0
                else "NaN"
            )
            avg_tle_vs_triton = avg_triton / avg_tle if avg_tle > 0 else 0
            print(f"{'Avg':>6}  {'-':>6}  {'-':>4}  {'-':>4}  {'-':>4}  "
                  f"{avg_vllm:10.4f}  {format_val(avg_flashinfer)}  {format_val(avg_tilelang, _TILELANG_MS_COL_WIDTH)}  {avg_triton:10.4f}  {avg_tle:10.4f}  "
                  f"{format_tflops(avg_vllm_tflops)}  {format_tflops(avg_flashinfer_tflops)}  {format_tflops(avg_tilelang_tflops, _TILELANG_TFLOPS_COL_WIDTH)}  {format_tflops(avg_tle_tflops)}  "
                  f"{format_ratio(avg_flashinfer_vs_vllm)}  {format_ratio(avg_tilelang_vs_vllm)}  {format_ratio(avg_triton_vs_vllm)}  "
                  f"{format_ratio(avg_tle_vs_vllm)}  {format_ratio(avg_tle_vs_flashinfer)}  {format_ratio(avg_tle_vs_tilelang)}  {format_ratio(avg_tle_vs_triton)}")

    _write_full_summary(
        ns,
        all_results,
        errors,
        include_triton=not ns.skip_triton,
    )

    if errors:
        print(f"\n  Failed: {errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
