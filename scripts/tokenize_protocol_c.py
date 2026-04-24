#!/usr/bin/env python
"""Tokenize FinePhrase + FineWeb-Edu-HQ for Protocol C (FinePhrase replication).

Two source datasets, both Llama-3.2 tokenized into a single output folder so
nanotron can read them as one mixed dataset (datatrove .ds shards). The mix
is achieved by writing both side-by-side into the same output directory; the
training-time data loader shuffles globally, producing a roughly 50/50 mix at
the configured token budget.

Source 1 — FinePhrase (synthetic, our target curve to reproduce)
    Reads:  data/reference/finephrase_table/table/*.parquet
    Field:  rollout_results[0]['text']     ← synthetic rephrase, NOT raw web
    Filter: skip rows where rollout_results is empty / has finish_reason!='stop'

Source 2 — FineWeb-Edu (mix-in baseline, FinePhrase paper's choice was DCLM)
    Reads:  data/reference/fineweb_edu_for_protC/data/CC-MAIN-*/*.parquet
    Field:  text
    Filter: int_score >= 3  (FineWeb-Edu-HQ partition)

Output:
    data/datasets/finephrase_table_plus_fwedu_hq_llama32/
        <SOURCE>/                          (datatrove writes .ds shards)
        tokenize.log
        SOURCE_STATS.json                  (token counts per source)

Usage:
    python scripts/tokenize_protocol_c.py            # both sources
    python scripts/tokenize_protocol_c.py --only finephrase
    python scripts/tokenize_protocol_c.py --only fineweb
    python scripts/tokenize_protocol_c.py --num-workers 32

Tokenizer: meta-llama/Llama-3.2-1B (vocab 128,256). Matches Protocol C.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from incepedia.config import DATASETS_DIR, REFERENCE_DIR

OUTPUT_ROOT = DATASETS_DIR / "finephrase_table_plus_fwedu_hq_llama32"
# hynky/Llama-3.2-1B-no-bos = same Llama-3.2 tokenizer (vocab 128,256) but
# without auto-BOS injection — the exact variant FinePhrase uses, and it's
# public (vs gated meta-llama/Llama-3.2-1B which requires HF approval).
TOKENIZER = "hynky/Llama-3.2-1B-no-bos"


def _log(out_root: Path, msg: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with (out_root / "tokenize.log").open("a") as f:
        f.write(line + "\n")


def tokenize_finephrase(num_workers: int) -> int:
    """Tokenize FinePhrase 'table' synthetic outputs."""
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.tokens import DocumentTokenizer

    src = REFERENCE_DIR / "finephrase_table" / "table"
    out = OUTPUT_ROOT / "finephrase"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {src}")
    _log(OUTPUT_ROOT, f"[finephrase] {len(files)} shards under {src}")
    _log(OUTPUT_ROOT, f"[finephrase] tokenizer={TOKENIZER} workers={num_workers}")
    _log(OUTPUT_ROOT, f"[finephrase] output={out}")

    # Custom adapter: pull the *generated* text out of the rollout_results column.
    # The column is a list of dicts: [{finish_reason, text, usage}, ...].
    # We take the first rollout (typical FinePhrase practice; only 1 sample per doc).
    # NOTE: datatrove ParquetReader requires the adapter to return a dict;
    # returning None crashes. Emit {"text": ""} for rows we want to skip;
    # base reader checks `if not parsed_data.get("text", None): return None`
    # and drops them before tokenization.
    EMPTY = {"text": "", "id": "", "metadata": {}}
    def adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
        rr = data.get("rollout_results")
        if not rr:
            return EMPTY
        first = rr[0] if isinstance(rr, list) and rr else None
        if not first:
            return EMPTY
        if first.get("finish_reason") not in (None, "stop"):
            return EMPTY
        text = first.get("text", "")
        if not text or len(text) < 50:
            return EMPTY
        return {
            "text": text,
            "id": str(data.get("id", f"{path}#{id_in_file}")),
            "metadata": {
                "src_dataset": "finephrase_table",
                "src_url": data.get("url"),
                "completion_tokens": first.get("usage", {}).get("completion_tokens"),
            },
        }

    pipeline = [
        ParquetReader(
            data_folder=str(src),
            glob_pattern="*.parquet",
            text_key="text",  # placeholder, our adapter overrides
            adapter=adapter,
        ),
        DocumentTokenizer(
            output_folder=str(out),
            tokenizer_name_or_path=TOKENIZER,
            eos_token="<|end_of_text|>",
            shuffle_documents=False,        # API renamed in current datatrove
            max_tokens_per_file=int(1e9),   # ~1B token shards
            save_filename="finephrase",
        ),
    ]
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=num_workers,
        workers=num_workers,
        logging_dir=str(out / "logs"),
    )
    t0 = time.time()
    executor.run()
    dt = time.time() - t0
    _log(OUTPUT_ROOT, f"[finephrase] done in {dt/60:.1f} min")
    return 0


def tokenize_fineweb(num_workers: int) -> int:
    """Tokenize FineWeb-Edu (HQ subset, int_score >= 3)."""
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.tokens import DocumentTokenizer

    src = REFERENCE_DIR / "fineweb_edu_for_protC" / "data"
    out = OUTPUT_ROOT / "fineweb_edu_hq"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {src}")
    _log(OUTPUT_ROOT, f"[fineweb] {len(files)} shards under {src}")
    _log(OUTPUT_ROOT, f"[fineweb] tokenizer={TOKENIZER} workers={num_workers}")
    _log(OUTPUT_ROOT, f"[fineweb] output={out}")

    EMPTY = {"text": "", "id": "", "metadata": {}}
    def adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
        if int(data.get("int_score", 0)) < 3:
            return EMPTY
        text = data.get("text", "")
        if not text or len(text) < 100:
            return EMPTY
        return {
            "text": text,
            "id": str(data.get("id", f"{path}#{id_in_file}")),
            "metadata": {
                "src_dataset": "fineweb_edu_hq",
                "url": data.get("url"),
                "int_score": data.get("int_score"),
            },
        }

    pipeline = [
        ParquetReader(
            data_folder=str(src),
            glob_pattern="**/*.parquet",
            text_key="text",
            adapter=adapter,
        ),
        DocumentTokenizer(
            output_folder=str(out),
            tokenizer_name_or_path=TOKENIZER,
            eos_token="<|end_of_text|>",
            shuffle_documents=False,
            max_tokens_per_file=int(1e9),
            save_filename="fineweb_edu_hq",
        ),
    ]
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=num_workers,
        workers=num_workers,
        logging_dir=str(out / "logs"),
    )
    t0 = time.time()
    executor.run()
    dt = time.time() - t0
    _log(OUTPUT_ROOT, f"[fineweb] done in {dt/60:.1f} min")
    return 0


def write_stats() -> None:
    """Best-effort token count summary by walking output .ds shards."""
    stats: dict = {"sources": {}}
    for source_dir in OUTPUT_ROOT.iterdir():
        if not source_dir.is_dir() or source_dir.name == "logs":
            continue
        ds_files = sorted(source_dir.glob("*.ds"))
        idx_files = sorted(source_dir.glob("*.ds.index"))
        if not ds_files:
            continue
        # rough token count = sum of .ds file sizes / 4 bytes (uint32 token ids)
        total_bytes = sum(p.stat().st_size for p in ds_files)
        approx_tokens = total_bytes // 4
        stats["sources"][source_dir.name] = {
            "ds_shards": len(ds_files),
            "index_files": len(idx_files),
            "total_bytes": total_bytes,
            "approx_tokens": approx_tokens,
        }
    out_path = OUTPUT_ROOT / "SOURCE_STATS.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"[stats] {out_path}")
    print(json.dumps(stats, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["finephrase", "fineweb"], default=None,
                   help="run only one source")
    p.add_argument("--num-workers", type=int, default=16,
                   help="parallel datatrove workers (each is a Python process)")
    args = p.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.only != "fineweb":
        tokenize_finephrase(args.num_workers)
    if args.only != "finephrase":
        tokenize_fineweb(args.num_workers)
    write_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
