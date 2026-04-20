#!/usr/bin/env python
"""Fetch reference datasets for Track 1/2 ablations.

Supports three targets (all public, no HF token required):

    cosmopedia-v2      — Track 1 (pure corpus)  & Track 2 (cooldown data)
                         Source: HuggingFaceTB/smollm-corpus, config "cosmopedia-v2"
                         ~28B tokens, ~70 GB parquet
                         Downloaded in full via snapshot_download.

    fineweb-edu-dedup  — Track 2 backbone (FineWeb-Edu deduplicated)
                         Source: HuggingFaceTB/smollm-corpus, subdir "fineweb-edu-dedup/"
                         Full set is 234 parquet shards, ~220B tokens / ~470 GB.
                         For Track 2 backbone we only need ~30B tokens → pull the
                         first N shards (default 32 → ~30B tokens).

    cosmopedia-v1      — Track 1 reference (old Cosmo-1B era)
                         Source: HuggingFaceTB/cosmopedia (standalone dataset)
                         ~25B tokens, ~92 GB parquet.
                         Downloaded in full via snapshot_download.

Usage
-----
    python scripts/fetch_reference_data.py cosmopedia-v2
    python scripts/fetch_reference_data.py fineweb-edu-dedup --num-shards 32
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

# FineWeb-Edu-dedup on smollm-corpus has 234 shards. 32 of them ≈ 30B tokens,
# which covers the Track 2 backbone (20B) + slack.
FINEWEB_TOTAL_SHARDS = 234
DEFAULT_FINEWEB_NUM_SHARDS = 32


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


# ── FineWeb-Edu: snapshot subset of upstream shards ─────────────────────

def fetch_fineweb_edu_shards(dest: Path, num_shards: int, total_shards: int = FINEWEB_TOTAL_SHARDS) -> None:
    """Pull the first `num_shards` of fineweb-edu-dedup via snapshot_download.

    Keeping upstream shard layout (`train-NNNNN-of-MMMMM.parquet`) means:
      - We preserve the dataset's own partitioning
      - Resume-on-failure is native
      - Downstream tokenizer can use `datasets.load_dataset` with the same file set
    """
    if num_shards > total_shards:
        raise ValueError(f"num_shards={num_shards} > total {total_shards}")
    patterns = [f"fineweb-edu-dedup/train-{i:05d}-of-{total_shards:05d}.parquet" for i in range(num_shards)]
    patterns.append("README.md")
    fetch_snapshot(
        repo_id="HuggingFaceTB/smollm-corpus",
        dest=dest,
        allow_patterns=patterns,
    )


# ── Fetch plans ──────────────────────────────────────────────────────────

def _plan_cosmopedia_v2(dest: Path) -> None:
    fetch_snapshot(
        repo_id="HuggingFaceTB/smollm-corpus",
        dest=dest,
        allow_patterns=["cosmopedia-v2/*", "README.md"],
    )


def _plan_fineweb_edu_dedup(dest: Path, num_shards: int) -> None:
    fetch_fineweb_edu_shards(dest=dest, num_shards=num_shards)


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
        "--num-shards",
        type=int,
        default=DEFAULT_FINEWEB_NUM_SHARDS,
        help=f"Number of fineweb-edu-dedup shards to fetch (of {FINEWEB_TOTAL_SHARDS}). "
        f"Default {DEFAULT_FINEWEB_NUM_SHARDS} ≈ 30B tokens.",
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
        _plan_fineweb_edu_dedup(dest, args.num_shards)
    elif args.dataset == "cosmopedia-v1":
        _plan_cosmopedia_v1(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
