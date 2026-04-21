#!/usr/bin/env python
"""One-shot descriptive stats for the Cosmopedia v2 parquet reference data.

Purpose
-------
Track 1 Protocol B baseline (`exp_ref_cosmopedia_v2_qwen3_seed42`) trains on
100% Cosmopedia v2.  The training pipeline does NOT re-weight or re-balance
Cosmopedia's internal categories — shards are tokenized with
`shuffle_documents=False`, so the subtype proportions that the model sees
simply match the upstream parquet mix.

This script prints that upstream mix (format / audience / token_length) so the
numbers can be attached as an appendix to the experiment record.  It is a
one-time analytical utility; it reads parquet directly and writes a tiny JSON
report.  It does NOT modify any training data.

Usage
-----
    python scripts/stats_cosmopedia_v2.py
    python scripts/stats_cosmopedia_v2.py --sample-rows 2_000_000  # faster sample
    python scripts/stats_cosmopedia_v2.py --out data/reference/cosmopedia_v2_stats.json

Outputs
-------
- stdout : human-readable breakdown
- JSON   : `data/reference/cosmopedia_v2_stats.json` (default) with
           { "n_rows": int,
             "format":   {cat: {count, share}},
             "audience": {cat: {count, share}},
             "token_length": {mean, p50, p90, p99, max} }
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from incepedia.config import REFERENCE_DIR


DEFAULT_SRC = REFERENCE_DIR / "cosmopedia_v2" / "cosmopedia-v2"
DEFAULT_OUT = REFERENCE_DIR / "cosmopedia_v2_stats.json"

# Columns we want to summarise.  We deliberately avoid reading `text` / `prompt`
# to keep memory + IO low; we only need the small metadata columns.
SUMMARY_COLUMNS = ["format", "audience", "token_length"]


def percentile(sorted_vals: list[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, int(round(pct * (len(sorted_vals) - 1)))))
    return int(sorted_vals[k])


def analyze(src: Path, sample_rows: int | None) -> dict:
    shards = sorted(src.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {src}")
    print(f"[stats] {len(shards)} shards under {src}", file=sys.stderr)

    fmt_cnt: Counter[str] = Counter()
    aud_cnt: Counter[str] = Counter()
    token_lengths: list[int] = []
    n_rows = 0

    for i, shard in enumerate(shards):
        # Only read the small metadata columns.
        table = pq.read_table(shard, columns=SUMMARY_COLUMNS)
        fmts = table.column("format").to_pylist()
        auds = table.column("audience").to_pylist()
        toks = table.column("token_length").to_pylist()
        fmt_cnt.update(fmts)
        aud_cnt.update(auds)
        token_lengths.extend(toks)
        n_rows += len(fmts)
        if (i + 1) % 10 == 0 or (i + 1) == len(shards):
            print(f"[stats] scanned {i+1}/{len(shards)} shards  rows={n_rows:,}",
                  file=sys.stderr)
        if sample_rows is not None and n_rows >= sample_rows:
            print(f"[stats] reached sample budget {sample_rows:,}; stopping early",
                  file=sys.stderr)
            break

    token_lengths.sort()
    tl_stats = {
        "mean": (sum(token_lengths) / len(token_lengths)) if token_lengths else 0.0,
        "p50": percentile(token_lengths, 0.50),
        "p90": percentile(token_lengths, 0.90),
        "p99": percentile(token_lengths, 0.99),
        "max": token_lengths[-1] if token_lengths else 0,
    }

    def _dist(c: Counter[str]) -> dict:
        total = sum(c.values()) or 1
        return {
            k: {"count": v, "share": round(v / total, 6)}
            for k, v in c.most_common()
        }

    return {
        "source": str(src),
        "n_rows_scanned": n_rows,
        "format": _dist(fmt_cnt),
        "audience": _dist(aud_cnt),
        "token_length": tl_stats,
    }


def _print_report(report: dict) -> None:
    print(f"\nscanned rows: {report['n_rows_scanned']:,}")
    for col in ("format", "audience"):
        print(f"\n── {col} ──")
        for k, v in report[col].items():
            print(f"  {k:<24} {v['count']:>12,}   {v['share']*100:6.2f}%")
    tl = report["token_length"]
    print("\n── token_length ──")
    for k in ("mean", "p50", "p90", "p99", "max"):
        print(f"  {k:<4}: {tl[k]:,}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help=f"parquet source dir (default: {DEFAULT_SRC})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"JSON output path (default: {DEFAULT_OUT})")
    p.add_argument("--sample-rows", type=int, default=None,
                   help="stop after N rows (for a quick sample); default: full scan")
    args = p.parse_args()

    report = analyze(args.src, sample_rows=args.sample_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    _print_report(report)
    print(f"\n[stats] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
