#!/usr/bin/env python
"""Fetch reference datasets for Track 1/2 ablations.

Supports three targets (all public, no HF token required):

    cosmopedia-v2      — Track 1 (pure corpus)  & Track 2 (cooldown data)
                         Source: HuggingFaceTB/smollm-corpus, config "cosmopedia-v2"
                         ~28B tokens, ~70 GB parquet
                         Downloaded in full via snapshot_download.

    fineweb-edu-dedup  — Track 2 backbone (FineWeb-Edu deduplicated)
                         Source: HuggingFaceTB/smollm-corpus, config "fineweb-edu-dedup"
                         Full set is ~220B tokens / ~470 GB.
                         We only need ~30B tokens → streaming mode with a token budget.

    cosmopedia-v1      — Track 1 reference (old Cosmo-1B era)
                         Source: HuggingFaceTB/cosmopedia (standalone dataset)
                         ~25B tokens, ~92 GB parquet.
                         Downloaded in full via snapshot_download.

Usage
-----
    python scripts/fetch_reference_data.py cosmopedia-v2
    python scripts/fetch_reference_data.py fineweb-edu-dedup --max-tokens 30_000_000_000
    python scripts/fetch_reference_data.py cosmopedia-v1

By default destinations are under  data/reference/<dataset-id>/  in the repo root.
All runs are resume-safe (snapshot_download checksums; streaming mode respects
a .fetch_state.json file).

Long downloads: launch with `&` or `nohup` to run in background; watch progress
via the written log file at  data/reference/<dataset-id>/fetch.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from incepedia.config import REFERENCE_DIR

# Enable HF's high-speed downloader if installed (no-op otherwise).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Per-row-approximate storage cost cache, in case we need to pre-allocate.
# (Not currently used, kept as reference for future planning.)

DEFAULT_FINEWEB_MAX_TOKENS = 30_000_000_000  # 30B — covers 20B backbone + slack


def _log(msg: str, dest: Path) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a") as f:
        f.write(line + "\n")


# ── Snapshot-based fetchers (full dataset, keep upstream shard layout) ──

def fetch_snapshot(
    repo_id: str,
    dest: Path,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    """Full repo / subdir download via huggingface_hub.snapshot_download.

    Idempotent: resumes partial downloads, skips completed files.
    """
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    log_path = dest / "fetch.log"
    _log(f"snapshot_download repo={repo_id} dest={dest}", log_path)
    _log(f"  allow_patterns={allow_patterns}  ignore_patterns={ignore_patterns}", log_path)

    t0 = time.time()
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=16,
    )
    dt = time.time() - t0
    total_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    _log(f"  done in {dt:.1f}s; {total_bytes/2**30:.2f} GiB on disk at {path}", log_path)


# ── Streaming-based fetcher (token-budgeted subset) ─────────────────────

def fetch_streaming_token_budget(
    repo_id: str,
    config_name: str,
    dest: Path,
    max_tokens: int,
    shard_rows: int = 100_000,
    token_count_col: str = "token_count",
) -> None:
    """Stream a dataset, stop once token budget hit, write local Parquet shards.

    State file `<dest>/.fetch_state.json` tracks progress; restart is safe-ish
    (we resume by skipping the rows already consumed in sequence, using HF's
    built-in streaming iteration ordering).
    """
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest.mkdir(parents=True, exist_ok=True)
    log_path = dest / "fetch.log"
    state_path = dest / ".fetch_state.json"

    state = {"tokens_written": 0, "rows_written": 0, "shard_idx": 0}
    if state_path.exists():
        state.update(json.loads(state_path.read_text()))
        _log(f"resuming at tokens={state['tokens_written']:,} shard={state['shard_idx']}", log_path)

    skip_rows = state["rows_written"]

    _log(f"streaming repo={repo_id} config={config_name} budget={max_tokens:,} tokens", log_path)
    ds = load_dataset(repo_id, config_name, split="train", streaming=True)

    buffer: list[dict] = []
    seen = 0
    tokens_in_buffer = 0
    t0 = time.time()

    for row in ds:
        seen += 1
        if seen <= skip_rows:
            continue
        tok = int(row.get(token_count_col, 0)) if row.get(token_count_col) else 0
        buffer.append(row)
        tokens_in_buffer += tok

        if len(buffer) >= shard_rows:
            _flush_parquet_shard(buffer, dest, state["shard_idx"])
            state["shard_idx"] += 1
            state["rows_written"] += len(buffer)
            state["tokens_written"] += tokens_in_buffer
            state_path.write_text(json.dumps(state, indent=2))
            elapsed = time.time() - t0
            _log(
                f"shard {state['shard_idx']-1:04d} written "
                f"(rows={state['rows_written']:,} tokens={state['tokens_written']:,} "
                f"elapsed={elapsed:.0f}s)",
                log_path,
            )
            buffer = []
            tokens_in_buffer = 0

        if state["tokens_written"] + tokens_in_buffer >= max_tokens:
            break

    if buffer:
        _flush_parquet_shard(buffer, dest, state["shard_idx"])
        state["shard_idx"] += 1
        state["rows_written"] += len(buffer)
        state["tokens_written"] += tokens_in_buffer
        state_path.write_text(json.dumps(state, indent=2))

    _log(
        f"done: wrote {state['rows_written']:,} rows / {state['tokens_written']:,} tokens "
        f"across {state['shard_idx']} shards in {time.time()-t0:.1f}s",
        log_path,
    )


def _flush_parquet_shard(rows: list[dict], dest: Path, idx: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    out = dest / f"shard-{idx:05d}.parquet"
    pq.write_table(table, out, compression="zstd")


# ── Fetch plans ──────────────────────────────────────────────────────────

def _plan_cosmopedia_v2(dest: Path) -> None:
    fetch_snapshot(
        repo_id="HuggingFaceTB/smollm-corpus",
        dest=dest,
        allow_patterns=["cosmopedia-v2/*", "README.md"],
    )


def _plan_fineweb_edu_dedup(dest: Path, max_tokens: int) -> None:
    fetch_streaming_token_budget(
        repo_id="HuggingFaceTB/smollm-corpus",
        config_name="fineweb-edu-dedup",
        dest=dest,
        max_tokens=max_tokens,
    )


def _plan_cosmopedia_v1(dest: Path) -> None:
    fetch_snapshot(
        repo_id="HuggingFaceTB/cosmopedia",
        dest=dest,
    )


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "dataset",
        choices=["cosmopedia-v2", "fineweb-edu-dedup", "cosmopedia-v1"],
        help="Which reference dataset to fetch.",
    )
    parser.add_argument("--dest", type=Path, default=None, help="Override destination path.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_FINEWEB_MAX_TOKENS,
        help="Token budget for streaming datasets (fineweb-edu-dedup). Default 30B.",
    )
    args = parser.parse_args()

    default_names = {
        "cosmopedia-v2": "cosmopedia_v2",
        "fineweb-edu-dedup": "fineweb_edu_dedup",
        "cosmopedia-v1": "cosmopedia_v1",
    }
    dest = args.dest or (REFERENCE_DIR / default_names[args.dataset])

    if args.dataset == "cosmopedia-v2":
        _plan_cosmopedia_v2(dest)
    elif args.dataset == "fineweb-edu-dedup":
        _plan_fineweb_edu_dedup(dest, args.max_tokens)
    elif args.dataset == "cosmopedia-v1":
        _plan_cosmopedia_v1(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
