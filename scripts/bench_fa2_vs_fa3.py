#!/usr/bin/env python
"""Micro-benchmark + numerical equivalence test: FA2 vs FA3.

This bypasses nanotron entirely.  We construct a Q/K/V tensor matching the
Qwen3-1.7B attention layer (heads=16, kv_heads=8, head_dim=128, seq=2048,
batch=1) and call both `flash_attn.flash_attn_interface.flash_attn_func` (FA2)
and `flash_attn_interface.flash_attn_func` (FA3) with identical inputs.

Outputs:
  - max_abs_diff between FA2 and FA3 outputs (target: ≤ 1e-2 in bf16)
  - throughput per backend (tokens/sec)
  - verdict (PASS / FAIL) for T1.

Usage:
  python scripts/bench_fa2_vs_fa3.py
  python scripts/bench_fa2_vs_fa3.py --batch 4 --seq 2048 --warmup 10 --iters 50
  python scripts/bench_fa2_vs_fa3.py --gpu 0      # pin to a specific GPU
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def make_qkv(batch: int, seq: int, n_q: int, n_kv: int, head_dim: int,
             dtype: torch.dtype, device: str, seed: int = 42):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, seq, n_q, head_dim, dtype=dtype, device=device, generator=g)
    k = torch.randn(batch, seq, n_kv, head_dim, dtype=dtype, device=device, generator=g)
    v = torch.randn(batch, seq, n_kv, head_dim, dtype=dtype, device=device, generator=g)
    return q, k, v


def time_fn(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--n_q", type=int, default=16)
    ap.add_argument("--n_kv", type=int, default=8)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--causal", action="store_true", default=True)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    free_mem, total_mem = torch.cuda.mem_get_info(args.gpu)
    print(f"[bench] gpu{args.gpu}  free={free_mem/2**30:.1f}GiB / total={total_mem/2**30:.1f}GiB")
    if free_mem < 200 * 2**20:
        sys.exit("[bench] not enough free GPU memory (need ~200 MiB)")

    print(f"[bench] tensor: batch={args.batch} seq={args.seq} n_q={args.n_q} "
          f"n_kv={args.n_kv} head_dim={args.head_dim} dtype={args.dtype}")
    q, k, v = make_qkv(args.batch, args.seq, args.n_q, args.n_kv, args.head_dim,
                       dtype, device)

    # --- FA2 -------------------------------------------------------------
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as fa2
    except Exception as e:
        sys.exit(f"[bench] FA2 import failed: {e!r}")

    def call_fa2():
        return fa2(q, k, v, dropout_p=0.0, causal=args.causal)

    out_fa2 = call_fa2()
    sec_fa2 = time_fn(call_fa2, args.warmup, args.iters)
    print(f"[bench] FA2  out shape={tuple(out_fa2.shape)}  per-call={sec_fa2*1000:.3f} ms")

    # --- FA3 -------------------------------------------------------------
    try:
        from flash_attn_interface import flash_attn_func as fa3
    except Exception as e:
        sys.exit(f"[bench] FA3 import failed: {e!r}")

    def call_fa3():
        out = fa3(q, k, v, causal=args.causal)
        return out[0] if isinstance(out, tuple) else out

    try:
        out_fa3 = call_fa3()
    except Exception as e:
        sys.exit(f"[bench] FA3 call failed (Hopper-only?): {e!r}")
    sec_fa3 = time_fn(call_fa3, args.warmup, args.iters)
    print(f"[bench] FA3  out shape={tuple(out_fa3.shape)}  per-call={sec_fa3*1000:.3f} ms")

    # --- Equivalence -----------------------------------------------------
    diff = (out_fa2.float() - out_fa3.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        out_fa2.float().flatten(), out_fa3.float().flatten(), dim=0
    ).item()
    speedup = sec_fa2 / sec_fa3 if sec_fa3 > 0 else float("inf")

    print()
    print("── results ─────────────────────────────────────────────")
    print(f"  max |FA2 - FA3|   = {max_diff:.2e}   (bf16 noise floor ~1e-3)")
    print(f"  mean |FA2 - FA3|  = {mean_diff:.2e}")
    print(f"  cosine similarity = {cosine:.6f}")
    print(f"  per-call FA2      = {sec_fa2*1000:.3f} ms")
    print(f"  per-call FA3      = {sec_fa3*1000:.3f} ms")
    print(f"  speedup (FA2/FA3) = {speedup:.2f}×")
    print()

    # T1 verdict criteria (set conservatively for bf16):
    #   - numerical equivalence: max_diff ≤ 1e-2 AND cosine ≥ 0.999
    #   - throughput: FA3 ≥ 1.10 × FA2 (10% target)
    pass_num = (max_diff <= 1e-2) and (cosine >= 0.999)
    pass_speed = speedup >= 1.10

    if pass_num and pass_speed:
        print("VERDICT: ✅ PASS — FA3 is numerically equivalent AND ≥10% faster")
        return 0
    elif pass_num:
        print(f"VERDICT: ⚠ NUM-OK / SPEED-LOW (only {speedup:.2f}× — recommend "
              f"keeping FA2 default until tested at full nanotron context)")
        return 2
    else:
        print(f"VERDICT: ❌ FAIL — FA3 differs from FA2 too much "
              f"(max_diff={max_diff:.2e}); keep FA2 default")
        return 1


if __name__ == "__main__":
    sys.exit(main())
