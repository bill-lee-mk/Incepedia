#!/usr/bin/env python
"""Pre-download all early-signal eval datasets into the isolated Incepedia HF cache.

Why this exists
---------------
When training eval kicks off with max_samples=1000 and 57 MMLU subsets,
lighteval spawns parallel HF Hub API calls that easily trip `429 Too Many
Requests` (unauthenticated limit ≈ 100 req/min).

This script serialises all dataset downloads through ONE worker with
configurable throttling, so the shared isolated cache (`data/hf_cache/`)
has everything ready BEFORE any eval runs.

Usage
-----
    python scripts/prefetch_eval_datasets.py                     # all early-signal
    python scripts/prefetch_eval_datasets.py --only csr-only     # just CSR (fast)
    python scripts/prefetch_eval_datasets.py --sleep 3           # 3s between reqs

Idempotent: already-cached datasets are skipped with a quick check.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from incepedia.config import DATA_DIR

# Isolated cache (SAME as the one used in runner's driver template)
HF_CACHE_ROOT = DATA_DIR / "hf_cache"
HF_DATASETS_CACHE = str(HF_CACHE_ROOT / "datasets")
HF_HOME = str(HF_CACHE_ROOT / "hf_home")

os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"


# ── Dataset inventory ────────────────────────────────────────────────────
# Each entry: (hf_repo, subset_list, [splits])
# `splits=None` means default; otherwise lighteval uses these evaluation splits.

CSR_DATASETS: list[tuple[str, list, list]] = [
    ("Rowan/hellaswag", ["default"], ["validation"]),
    ("allenai/winogrande", ["winogrande_xl"], ["validation"]),
    ("ybisk/piqa", ["plain_text"], ["validation"]),
    ("lighteval/siqa", ["default"], ["validation"]),
    ("allenai/openbookqa", ["main"], ["test"]),
    ("allenai/ai2_arc", ["ARC-Easy", "ARC-Challenge"], ["test"]),
    ("tau/commonsense_qa", ["default"], ["validation"]),
    ("google/boolq", ["default"], ["validation"]),
    ("mandarjoshi/trivia_qa", ["rc.nocontext"], ["validation"]),
    ("TIGER-Lab/MMLU-Pro", ["default"], ["test", "validation"]),
]

# MMLU 57 subsets
MMLU_SUBSETS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]
MMLU_DATASETS: list[tuple[str, list, list]] = [
    ("lighteval/mmlu", MMLU_SUBSETS, ["test", "dev"]),
    ("TIGER-Lab/MMLU-STEM", ["default"], ["test"]),
]

# GSM8K is the generation-heavy anchor in the cosmopedia-full eval suite.
# lighteval needs both `train` (for few-shot sampling) and `test` (for eval).
GENERATION_DATASETS: list[tuple[str, list, list]] = [
    ("openai/gsm8k", ["main"], ["train", "test"]),
]


def prefetch(repo: str, subsets: list, splits: list | None, sleep: float) -> int:
    """Download a single HF dataset repo (with all subsets). Returns # subsets downloaded."""
    from datasets import load_dataset
    n = 0
    for subset in subsets:
        try:
            print(f"  [{repo}] subset={subset} ...", flush=True)
            t0 = time.time()
            _ = load_dataset(
                repo,
                subset if subset != "default" else None,
                trust_remote_code=True,
            )
            dt = time.time() - t0
            print(f"    ✅ cached in {dt:.1f}s", flush=True)
            n += 1
        except Exception as e:
            print(f"    ❌ FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
        time.sleep(sleep)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only",
                        choices=["csr-only", "mmlu-only", "generation-only",
                                 "early-signal", "cosmopedia-full"],
                        default="cosmopedia-full",
                        help="Which task group to prefetch (default: cosmopedia-full).")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds between requests to avoid HF rate limits (default: 2).")
    args = parser.parse_args()

    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[prefetch] cache dir: {HF_DATASETS_CACHE}")
    print(f"[prefetch] throttle : {args.sleep}s between requests")

    buckets: list[tuple[str, list]] = []
    if args.only in ("csr-only", "early-signal", "cosmopedia-full"):
        buckets.append(("CSR", CSR_DATASETS))
    if args.only in ("mmlu-only", "early-signal", "cosmopedia-full"):
        buckets.append(("MMLU", MMLU_DATASETS))
    if args.only in ("generation-only", "cosmopedia-full"):
        buckets.append(("GENERATION", GENERATION_DATASETS))

    total = 0
    t_start = time.time()
    for label, datasets in buckets:
        print(f"\n━━━━ {label} ({sum(len(s) for _, s, _ in datasets)} configs) ━━━━")
        for repo, subsets, splits in datasets:
            n = prefetch(repo, subsets, splits, sleep=args.sleep)
            total += n

    dt = time.time() - t_start
    print(f"\n[prefetch] done: {total} configs cached in {dt/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
