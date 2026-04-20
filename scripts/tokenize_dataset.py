#!/usr/bin/env python
"""Tokenize reference datasets into nanotron-compatible shards.

Uses datatrove's standard DocumentTokenizer pipeline: reads Parquet shards with
a `text` column, encodes with the Mistral-7B-v0.1 tokenizer (matching our
training config), and writes token-id shards for nanotron to consume.

Usage:
    python scripts/tokenize_dataset.py cosmopedia-v2
    python scripts/tokenize_dataset.py fineweb-edu-dedup
    python scripts/tokenize_dataset.py <id> --text-col content --num-workers 32

Inputs expected at:
    data/reference/<id>/              (Parquet shards from fetch_reference_data.py)

Outputs written to:
    data/datasets/<id>/tokenized/     (nanotron-format .ds shards + index)
    data/datasets/<id>/tokenize.log
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from incepedia.config import DATASETS_DIR, REFERENCE_DIR

# Default mapping for common datasets (auto-detects the right text column).
DATASET_SPEC: dict[str, dict] = {
    "cosmopedia-v2": dict(
        src_subdir="cosmopedia_v2/cosmopedia-v2",  # reflects HF repo layout
        out_id="cosmopedia_v2_reference",
        text_col="text",
        glob="*.parquet",
    ),
    "fineweb-edu-dedup": dict(
        src_subdir="fineweb_edu_dedup/fineweb-edu-dedup",
        out_id="fineweb_edu_backbone",
        text_col="text",
        glob="*.parquet",
    ),
    "cosmopedia-v1": dict(
        src_subdir="cosmopedia_v1",
        out_id="cosmopedia_v1_reference",
        text_col="text",
        glob="**/*.parquet",
    ),
}


def tokenize(
    source_dir: Path,
    output_dir: Path,
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
    text_col: str = "text",
    num_workers: int = 16,
    glob: str = "*.parquet",
    eos_token: str | None = "</s>",
) -> None:
    """Run datatrove pipeline: Parquet → tokenized shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tokenize.log"

    def _log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    # Count inputs
    files = sorted(source_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f"No {glob} under {source_dir}")
    _log(f"found {len(files)} shards under {source_dir}")
    _log(f"tokenizer={tokenizer_name} text_col={text_col} workers={num_workers}")
    _log(f"output_dir={output_dir}")

    # Lazy import — datatrove pulls heavy deps
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.tokens import DocumentTokenizer

    pipeline = [
        ParquetReader(
            data_folder=str(source_dir),
            glob_pattern=glob,
            text_key=text_col,
        ),
        DocumentTokenizer(
            output_folder=str(output_dir),
            tokenizer_name_or_path=tokenizer_name,
            eos_token=eos_token,
            shuffle_documents=False,          # preserve upstream order for determinism
            max_tokens_per_file=500_000_000,  # 500M-token shards
            save_index=True,
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=num_workers,
        workers=num_workers,
        logging_dir=str(output_dir / "datatrove_logs"),
    )
    t0 = time.time()
    executor.run()
    dt = time.time() - t0
    _log(f"done in {dt/60:.1f} min")

    # summarise
    ds_files = list(output_dir.glob("*.ds"))
    total_bytes = sum(p.stat().st_size for p in ds_files if p.is_file())
    _log(f"wrote {len(ds_files)} .ds shards, total {total_bytes/2**30:.2f} GiB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset id: cosmopedia-v2 | fineweb-edu-dedup | cosmopedia-v1 (or custom)")
    parser.add_argument("--src", type=Path, default=None, help="override source directory")
    parser.add_argument("--out", type=Path, default=None, help="override output directory")
    parser.add_argument("--text-col", default=None, help="override text column name")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--glob", default=None, help="override shard glob pattern")
    parser.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")
    args = parser.parse_args()

    spec = DATASET_SPEC.get(args.dataset, {})
    source_dir = args.src or (REFERENCE_DIR / spec.get("src_subdir", args.dataset))
    out_id = spec.get("out_id", args.dataset.replace("-", "_"))
    output_dir = args.out or (DATASETS_DIR / out_id / "tokenized")
    text_col = args.text_col or spec.get("text_col", "text")
    glob = args.glob or spec.get("glob", "*.parquet")

    tokenize(
        source_dir=source_dir,
        output_dir=output_dir,
        tokenizer_name=args.tokenizer,
        text_col=text_col,
        num_workers=args.num_workers,
        glob=glob,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
