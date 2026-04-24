#!/usr/bin/env python
"""Benchmark OpenRouter throughput at various concurrency levels.

For each concurrency C in [8, 16, 32, 64, 128, 256]:
  - Fire N requests concurrently (default 100)
  - Record actual rps, 429 count, error count, avg/p50/p95 latency, tok/s
Prints a ranked table so you can pick the sweet spot before the 1B run.

Safe cost: ~100 requests × 6 levels × ~800 tok ≈ 480k output tokens ≈ $0.5-1
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from incepedia.generation.client import OpenRouterClient


async def run_level(
    seeds_path: Path,
    model: str,
    prompt_template: str,
    concurrency: int,
    n_requests: int,
    max_tokens: int,
    temperature: float,
) -> dict:
    tbl = pq.read_table(seeds_path, columns=["id", "text"])
    ids = tbl.column("id").to_pylist()
    texts = tbl.column("text").to_pylist()
    # Take a random-ish slice (shifted per level to avoid cached prompts)
    start = (concurrency * 97) % max(len(ids) - n_requests, 1)
    ids = ids[start:start + n_requests]
    texts = texts[start:start + n_requests]

    client = OpenRouterClient(model=model, concurrency=concurrency, timeout_s=180.0, max_retries=3)
    t0 = time.time()
    async with client:
        tasks = [
            client.generate(
                doc_id=i,
                messages=[{"role": "user", "content": prompt_template.format(document=t)}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for i, t in zip(ids, texts)
        ]
        results = await asyncio.gather(*tasks)
    dur = time.time() - t0
    ok = [r for r in results if r.error is None and r.completion_tokens > 0]
    latencies = [r.latency_s for r in ok]
    total_out = sum(r.completion_tokens for r in ok)
    total_cost = sum(r.total_cost_usd for r in ok)
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "n_429": client.stats.n_429,
        "elapsed_s": round(dur, 1),
        "rps": round(len(ok) / dur, 2),
        "tok_out": total_out,
        "tok_per_s": round(total_out / dur, 1),
        "p50_latency_s": round(statistics.median(latencies), 2) if latencies else None,
        "p95_latency_s": round(
            statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies) if latencies else 0,
            2,
        ),
        "cost_usd": round(total_cost, 4),
    }


async def main_async(args: argparse.Namespace) -> None:
    prompt_path = Path(__file__).resolve().parent.parent / "src" / "incepedia" / "generation" / "prompts" / f"{args.prompt}.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    levels = [int(x) for x in args.levels.split(",")]
    rows = []
    print(f"[bench] model={args.model} prompt={args.prompt} n={args.n_requests} levels={levels}")
    for c in levels:
        print(f"\n[bench] === concurrency={c} ===")
        row = await run_level(
            seeds_path=args.seeds,
            model=args.model,
            prompt_template=prompt_template,
            concurrency=c,
            n_requests=args.n_requests,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"[bench] result: {row}")
        rows.append(row)
        # small rest between levels to let rate-limit windows clear
        await asyncio.sleep(3)

    print("\n" + "=" * 100)
    print(f"{'conc':>5}  {'n_ok':>5}  {'n_429':>6}  {'rps':>6}  {'tok/s':>7}  {'p50_lat':>8}  {'p95_lat':>8}  {'cost':>7}")
    print("=" * 100)
    for r in rows:
        print(
            f"{r['concurrency']:>5}  "
            f"{r['n_ok']:>5}  "
            f"{r['n_429']:>6}  "
            f"{r['rps']:>6.2f}  "
            f"{r['tok_per_s']:>7.1f}  "
            f"{str(r['p50_latency_s']):>8}  "
            f"{str(r['p95_latency_s']):>8}  "
            f"${r['cost_usd']:>6.3f}"
        )
    best = max(rows, key=lambda r: r["tok_per_s"])
    print(f"\n[bench] BEST by tok/s: concurrency={best['concurrency']} → {best['tok_per_s']} tok/s "
          f"(429={best['n_429']}, p95={best['p95_latency_s']}s)")
    total_cost = sum(r["cost_usd"] for r in rows)
    print(f"[bench] total benchmark cost: ${total_cost:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=Path, required=True, help="Parquet with columns [id, text]")
    p.add_argument("--model", default="deepseek/deepseek-chat")
    p.add_argument("--prompt", default="tutorial", help="prompt name under src/incepedia/generation/prompts/")
    p.add_argument("--levels", default="8,16,32,64,128,256")
    p.add_argument("--n-requests", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()
    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
