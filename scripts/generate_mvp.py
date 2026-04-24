#!/usr/bin/env python
"""Generate Incepedia MVP synthetic corpus from FineWeb-Edu seeds via OpenRouter.

Design:
  - Input: Parquet with [id, text, ...] seed columns
  - Process: one prompt template (configurable), one model (configurable)
  - Output: ./data/raw_generations/{batch_id}/part-NNNNNN.parquet (shard_rows=1000)
  - Checkpoint: ./data/raw_generations/{batch_id}/_processed_ids.txt (append-only)
  - Resume: skip any doc_id already in _processed_ids.txt on startup
  - Live progress: log every 100 completed generations (ok/err/429, rps, tok/s, cost, ETA)

Schema (C11):
  id, text, generator, prompt_tokens, completion_tokens, total_cost_usd,
  latency_s, temperature, attempts, seed_id, seed_hash, seed_url, task,
  prompt_template, batch_id, created_at, error

Usage:
  python scripts/generate_mvp.py --batch configs/batches/mvp_20260424.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from incepedia.generation.client import OpenRouterClient, GenerationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("incepedia.generate_mvp")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "src" / "incepedia" / "generation" / "prompts"


def load_batch_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    required = ["batch_id", "seeds_path", "output_dir", "model", "prompt",
                "concurrency", "shard_rows", "max_tokens", "temperature"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"batch.yaml missing required keys: {missing}")
    return cfg


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def load_processed_ids(ckpt_path: Path) -> set[str]:
    if not ckpt_path.exists():
        return set()
    with open(ckpt_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def build_row(
    seed_id: str,
    seed_text: str,
    seed_url: str | None,
    result: GenerationResult,
    cfg: dict,
) -> dict[str, Any]:
    seed_hash = hashlib.sha1(seed_text[:512].encode("utf-8")).hexdigest()[:16]
    return {
        "id": f"{cfg['batch_id']}-{seed_id}",
        "text": result.text,
        "generator": result.generator,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_cost_usd": float(result.total_cost_usd),
        "latency_s": float(result.latency_s),
        "temperature": float(result.temperature),
        "attempts": int(result.attempts),
        "seed_id": seed_id,
        "seed_hash": seed_hash,
        "seed_url": seed_url or "",
        "task": cfg["prompt"],
        "prompt_template": cfg["prompt"],
        "batch_id": cfg["batch_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "error": result.error or "",
    }


async def run_batch(cfg: dict) -> None:
    batch_id = cfg["batch_id"]
    out_dir = Path(cfg["output_dir"]) / batch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "_processed_ids.txt"
    log_path = out_dir / "generate.log"
    # also mirror to file
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    # Seeds
    seeds_tbl = pq.read_table(cfg["seeds_path"], columns=["id", "text", "url"] if "url" in pq.ParquetFile(cfg["seeds_path"]).schema_arrow.names else ["id", "text"])
    all_ids = seeds_tbl.column("id").to_pylist()
    all_texts = seeds_tbl.column("text").to_pylist()
    all_urls = seeds_tbl.column("url").to_pylist() if "url" in seeds_tbl.column_names else [None] * len(all_ids)

    processed = load_processed_ids(ckpt_path)
    log.info("[%s] seeds loaded: %d;  already processed: %d", batch_id, len(all_ids), len(processed))

    # filter to unprocessed
    todo = [(i, t, u) for i, t, u in zip(all_ids, all_texts, all_urls) if i not in processed]
    if cfg.get("max_requests"):
        todo = todo[: int(cfg["max_requests"])]
    log.info("[%s] will process: %d seeds", batch_id, len(todo))
    if not todo:
        log.info("[%s] nothing to do", batch_id)
        return

    # prompt template
    prompt_tpl = load_prompt(cfg["prompt"])

    client = OpenRouterClient(
        model=cfg["model"],
        concurrency=int(cfg["concurrency"]),
        timeout_s=float(cfg.get("timeout_s", 180)),
        max_retries=int(cfg.get("max_retries", 6)),
    )

    shard_rows = int(cfg["shard_rows"])
    max_cost = float(cfg.get("max_cost_usd", 0) or 0)

    shard_buffer: list[dict[str, Any]] = []
    shard_idx = len(list(out_dir.glob("part-*.parquet")))
    ckpt_fh = open(ckpt_path, "a", buffering=1)  # line-buffered

    async def process_one(seed_id: str, seed_text: str, seed_url: str | None) -> dict[str, Any] | None:
        messages = [{"role": "user", "content": prompt_tpl.format(document=seed_text)}]
        r = await client.generate(
            doc_id=seed_id,
            messages=messages,
            temperature=float(cfg["temperature"]),
            top_p=float(cfg.get("top_p", 0.95)),
            max_tokens=int(cfg["max_tokens"]),
        )
        if r.error is None and r.completion_tokens > 0:
            return build_row(seed_id, seed_text, seed_url, r, cfg)
        return None

    def flush_shard() -> None:
        nonlocal shard_buffer, shard_idx
        if not shard_buffer:
            return
        tmp = out_dir / f".part-{shard_idx:06d}.tmp.parquet"
        final = out_dir / f"part-{shard_idx:06d}.parquet"
        tbl = pa.Table.from_pylist(shard_buffer)
        pq.write_table(tbl, tmp, compression="zstd")
        tmp.rename(final)  # atomic
        log.info("[%s] wrote shard %d (%d rows) → %s", batch_id, shard_idx, len(shard_buffer), final.name)
        shard_idx += 1
        shard_buffer = []

    t_start = time.time()
    last_log = t_start
    done_count = 0
    async with client:
        # Use asyncio.as_completed via bounded batch submission
        BATCH = int(cfg["concurrency"]) * 4
        i = 0
        while i < len(todo):
            chunk = todo[i : i + BATCH]
            i += BATCH
            tasks = [
                asyncio.create_task(process_one(sid, stx, surl))
                for (sid, stx, surl) in chunk
            ]
            chunk_ids_in_order = [sid for (sid, _, _) in chunk]
            for fut, sid in zip(tasks, chunk_ids_in_order):
                try:
                    row = await fut
                except Exception as e:
                    log.exception("unexpected task error for %s: %s", sid, e)
                    row = None
                done_count += 1
                if row is not None:
                    shard_buffer.append(row)
                    ckpt_fh.write(sid + "\n")
                    if len(shard_buffer) >= shard_rows:
                        flush_shard()
                # progress log every 100
                if done_count % 100 == 0:
                    now = time.time()
                    s = client.stats.snapshot()
                    eta_s = (len(todo) - done_count) / max(s["rps"], 0.01)
                    log.info(
                        "[%s] progress %d/%d | ok=%d err=%d 429=%d | %.1f rps | %.0f tok/s | $%.3f | ETA %.1fh",
                        batch_id, done_count, len(todo), s["n_ok"], s["n_err"], s["n_429"],
                        s["rps"], s["tok_per_s"], s["cost_usd"], eta_s / 3600.0,
                    )
                    last_log = now
                # cost circuit breaker
                if max_cost > 0 and client.stats.total_cost_usd >= max_cost:
                    log.error("[%s] cost cap $%.2f reached; flushing and stopping", batch_id, max_cost)
                    flush_shard()
                    ckpt_fh.close()
                    return

    flush_shard()
    ckpt_fh.close()
    dur = time.time() - t_start
    s = client.stats.snapshot()
    log.info(
        "[%s] DONE. done=%d | ok=%d err=%d 429=%d | elapsed=%.1fh | %.1f rps | %.0f tok/s | $%.3f total",
        batch_id, done_count, s["n_ok"], s["n_err"], s["n_429"],
        dur / 3600, s["rps"], s["tok_per_s"], s["cost_usd"],
    )
    # write final summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "batch_id": batch_id,
        "config": cfg,
        "processed_docs": done_count,
        "stats": s,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    log.info("[%s] summary → %s", batch_id, summary_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=Path, required=True, help="path to batch yaml")
    args = p.parse_args()
    cfg = load_batch_config(args.batch)
    asyncio.run(run_batch(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
