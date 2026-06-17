#!/usr/bin/env python3
"""Single-case profile script for NCU: batch=1, s_kv=8192, h_q=128, d=576, dv=512."""

import math, os, sys, torch, triton
import flaggems_vllm

b, s_q, h_q, h_kv, d, dv = 1, 1, 128, 1, 576, 512
seqlen, block_size = 8192, 64
device = flaggems_vllm.device
dtype = torch.bfloat16

torch.manual_seed(42)
cache_seqlens = torch.tensor([seqlen], dtype=torch.int32, device=device)
max_seqlen_pad = triton.cdiv(seqlen, 256) * 256
q = torch.randn([b, s_q, h_q, d], dtype=dtype, device=device)
block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32, device=device).view(b, max_seqlen_pad // block_size)
blocked_k = torch.randn([block_table.numel(), block_size, h_kv, d], dtype=dtype, device=device)

impl = sys.argv[1] if len(sys.argv) > 1 else "tle"

if impl == "vllm":
    from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata
    def fn():
        sched_meta, _ = get_mla_metadata()
        flash_mla_with_kvcache(q, blocked_k, block_table, cache_seqlens, dv, sched_meta, None, softmax_scale=1/math.sqrt(d), causal=True)
elif impl == "tle":
    os.environ["FLAGGEMS_VLLM_FLASH_MLA_TLE"] = "1"
    os.environ["FLAGGEMS_VLLM_FLASH_MLA_TLE_VARIANT"] = "auto"
    def fn():
        flaggems_vllm.flash_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, True)

# warmup
for _ in range(3):
    fn()
torch.cuda.synchronize()

# profiled run
torch.cuda.cudart().cudaProfilerStart()
fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
