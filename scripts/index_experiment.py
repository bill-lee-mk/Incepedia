#!/usr/bin/env python
"""Incepedia · INDEX.parquet 维护器.

所有 experiments/exp_*/config.yaml 和 metrics.json 聚合到仓库根的 INDEX.parquet,
方便后续用 pandas / DuckDB 一行找到最佳实验、分叉、历史对比。

用法:
    python scripts/index_experiment.py rebuild       # 从零扫描,重建 INDEX.parquet
    python scripts/index_experiment.py add <exp_id>  # 增量追加一个实验
    python scripts/index_experiment.py show          # 打印当前索引 (pretty table)
    python scripts/index_experiment.py best <metric> # 按某指标倒排前 10
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
INDEX_PATH = REPO_ROOT / "INDEX.parquet"

SCHEMA_COLS = [
    "exp_id",
    "stage",
    "branch_from",
    "dataset_id",
    "dataset_tokens",
    "model_arch",
    "train_tokens",
    "ckpt_path",
    "config_hash",
    "config_path",
    "seed",
    "timestamp_utc",
    "notes",
    # early-signal metrics (filled as experiments complete)
    "hellaswag",
    "winogrande",
    "piqa",
    "siqa",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "commonsense_qa",
    "boolq",
    "mmlu_cloze",
    "mmlu_mc",
    "mmlu_pro_cloze",
    "trivia_qa",
    "gsm8k_5shot",
    "humaneval",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_exp(exp_dir: Path) -> dict | None:
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    row = {col: None for col in SCHEMA_COLS}
    row["exp_id"] = cfg.get("exp_id", exp_dir.name)
    row["stage"] = cfg.get("stage")
    row["branch_from"] = cfg.get("branch_from")
    row["dataset_id"] = (cfg.get("dataset") or {}).get("id")
    row["dataset_tokens"] = (cfg.get("dataset") or {}).get("tokens")
    row["model_arch"] = (cfg.get("model") or {}).get("arch")
    row["seed"] = (cfg.get("training") or {}).get("seed")
    row["notes"] = cfg.get("notes")
    row["config_path"] = str(cfg_path.relative_to(REPO_ROOT))
    row["config_hash"] = _sha256_file(cfg_path)

    ckpt_dir = exp_dir / "ckpt"
    if ckpt_dir.exists():
        row["ckpt_path"] = str(ckpt_dir.relative_to(REPO_ROOT))

    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            m = {}
        row["train_tokens"] = m.get("train_tokens")
        for col in SCHEMA_COLS:
            if col in m:
                row[col] = m[col]

    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row


def rebuild() -> pd.DataFrame:
    rows: list[dict] = []
    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
                continue
            row = _read_exp(exp_dir)
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows, columns=SCHEMA_COLS)
    df.to_parquet(INDEX_PATH, index=False)
    print(f"[rebuild] {len(df)} experiments indexed → {INDEX_PATH}")
    return df


def add(exp_id: str) -> None:
    exp_dir = EXPERIMENTS_DIR / exp_id
    row = _read_exp(exp_dir)
    if row is None:
        print(f"[err] no config.yaml for {exp_id}", file=sys.stderr)
        sys.exit(1)
    if INDEX_PATH.exists():
        df = pd.read_parquet(INDEX_PATH)
        df = df[df.exp_id != row["exp_id"]]
    else:
        df = pd.DataFrame(columns=SCHEMA_COLS)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(INDEX_PATH, index=False)
    print(f"[add] {exp_id} → INDEX.parquet ({len(df)} rows total)")


def show() -> None:
    if not INDEX_PATH.exists():
        print("(empty — no INDEX.parquet yet)")
        return
    df = pd.read_parquet(INDEX_PATH)
    if df.empty:
        print("(empty INDEX.parquet)")
        return
    cols = [c for c in ["exp_id", "stage", "dataset_id", "train_tokens",
                        "hellaswag", "mmlu_cloze", "gsm8k_5shot", "notes"] if c in df.columns]
    with pd.option_context("display.max_rows", 200, "display.max_colwidth", 60, "display.width", 180):
        print(df[cols].to_string(index=False))


def best(metric: str, top: int = 10) -> None:
    if not INDEX_PATH.exists():
        print("(empty — no INDEX.parquet yet)")
        return
    df = pd.read_parquet(INDEX_PATH)
    if metric not in df.columns:
        print(f"[err] unknown metric '{metric}'. options: {SCHEMA_COLS}", file=sys.stderr)
        sys.exit(1)
    df = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top)
    cols = [c for c in ["exp_id", "stage", "dataset_id", "train_tokens", metric, "notes"] if c in df.columns]
    print(df[cols].to_string(index=False))


def _usage():
    print(__doc__)
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        _usage()
    cmd = sys.argv[1]
    if cmd == "rebuild":
        rebuild()
    elif cmd == "add":
        if len(sys.argv) < 3:
            _usage()
        add(sys.argv[2])
    elif cmd == "show":
        show()
    elif cmd == "best":
        if len(sys.argv) < 3:
            _usage()
        best(sys.argv[2])
    else:
        _usage()


if __name__ == "__main__":
    main()
