#!/usr/bin/env python
"""Sample FineWeb-Edu seeds for MVP generation.

Filters:
  - language == 'en'
  - int_score >= 3  (fineweb-edu classifier edu score)
  - 300 <= token_count <= 2000  (avoid too short/too long docs)

Output:
  data/seeds/mvp_seeds.parquet with columns [id, text, token_count, int_score, url]
  Default target: 1.3M seeds (~1B output tokens at 800 avg completion)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from incepedia.config import REFERENCE_DIR, DATA_DIR

DEFAULT_SRC = REFERENCE_DIR / "fineweb_edu_dedup"
DEFAULT_OUT = DATA_DIR / "seeds" / "mvp_seeds.parquet"


def sample_seeds(
    src: Path,
    out: Path,
    n_target: int,
    min_score: int,
    min_tokens: int,
    max_tokens: int,
    seed: int,
) -> None:
    shards = sorted(src.glob("shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no shard-*.parquet under {src}")
    print(f"[seeds] scanning {len(shards)} shards under {src}")
    out.parent.mkdir(parents=True, exist_ok=True)

    kept_tables: list[pa.Table] = []
    kept = 0
    scanned = 0
    for i, shard in enumerate(shards):
        tbl = pq.read_table(
            shard,
            columns=["id", "text", "metadata"],
        )
        md = tbl.column("metadata").combine_chunks()
        # flatten struct access
        lang = pc.struct_field(md, "language")
        int_score = pc.struct_field(md, "int_score")
        token_count = pc.struct_field(md, "token_count")
        url = pc.struct_field(md, "url")
        mask = pc.and_(
            pc.equal(lang, "en"),
            pc.and_(
                pc.greater_equal(int_score, min_score),
                pc.and_(
                    pc.greater_equal(token_count, min_tokens),
                    pc.less_equal(token_count, max_tokens),
                ),
            ),
        )
        keep = tbl.append_column("int_score", int_score).append_column(
            "token_count", token_count
        ).append_column("url", url).filter(mask).drop_columns(["metadata"])
        kept_tables.append(keep)
        kept += keep.num_rows
        scanned += tbl.num_rows
        print(
            f"[seeds] shard {i+1}/{len(shards)}: scanned={tbl.num_rows:>7,d} "
            f"kept={keep.num_rows:>7,d}  cum_kept={kept:>8,d}  cum_scanned={scanned:>8,d}"
        )
        if kept >= n_target * 1.3:  # enough slack for shuffle
            break

    full = pa.concat_tables(kept_tables).combine_chunks()
    print(f"[seeds] total kept after filter: {full.num_rows:,}")
    if full.num_rows < n_target:
        print(f"[seeds] WARNING: kept {full.num_rows:,} < target {n_target:,}; using all kept")
        sampled = full
    else:
        import pyarrow.compute as _pc
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = rng.choice(full.num_rows, size=n_target, replace=False)
        idx.sort()  # preserve shard locality
        sampled = full.take(pa.array(idx))
    print(f"[seeds] writing {sampled.num_rows:,} seeds → {out}")
    pq.write_table(sampled, out, compression="zstd")
    print(f"[seeds] done. size={out.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-target", type=int, default=1_300_000)
    p.add_argument("--min-score", type=int, default=3)
    p.add_argument("--min-tokens", type=int, default=300)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    sample_seeds(args.src, args.out, args.n_target, args.min_score,
                 args.min_tokens, args.max_tokens, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
