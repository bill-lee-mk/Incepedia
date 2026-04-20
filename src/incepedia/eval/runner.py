"""Lighteval subprocess runner.

Wraps `accelerate launch lighteval/run_evals_accelerate.py` with a Python API
and aggregates per-task scores into a single metrics.json compatible with
INDEX.parquet (see scripts/index_experiment.py:SCHEMA_COLS).

Usage
-----
    from incepedia.eval.runner import EvalRunner

    runner = EvalRunner(
        model="HuggingFaceTB/cosmo-1b",
        output_dir="experiments/exp_ref_cosmopedia_v2/eval",
        task_group="early-signal",
        num_processes=8,
        batch_size=16,
    )
    scores = runner.run()
    runner.write_metrics_json("experiments/exp_ref_cosmopedia_v2/metrics.json")

Why a wrapper
-------------
- Ensures we always run with the *exact same* task string cosmopedia used,
  so scores are comparable to SmolLM / FineWeb published numbers.
- Handles lighteval's output-dir layout, which varies slightly across versions.
- Normalises task scores into the column names we maintain in INDEX.parquet.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from incepedia.config import REPO_ROOT
from incepedia.eval.lighteval_tasks import TASKS_GROUPS

# Map lighteval's task name → the column we use in INDEX.parquet.
# Keep in sync with scripts/index_experiment.py:SCHEMA_COLS.
_TASK_TO_COLUMN: dict[str, str] = {
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "piqa": "piqa",
    "siqa": "siqa",
    "openbookqa": "openbookqa",
    "arc:easy": "arc_easy",
    "arc:challenge": "arc_challenge",
    "commonsense_qa": "commonsense_qa",
    "boolq": "boolq",
    "mmlu_pro_cloze": "mmlu_pro_cloze",
    "mmlu_pro_mc": "mmlu_pro_mc",
    "trivia_qa": "trivia_qa",
    "gsm8k": "gsm8k_5shot",
}

# Which metric name inside a task's result dict to take as the canonical score.
# lighteval tasks expose multiple metrics (acc, acc_norm, ...); we take acc_norm
# when available (matches SmolLM / FineWeb reporting) else plain acc.
_METRIC_PREFERENCE = (
    "acc_norm_nospace",
    "acc_norm",
    "loglikelihood_acc_norm_nospace",
    "loglikelihood_acc_norm",
    "acc",
    "loglikelihood_acc",
    "quasi_exact_match",
    "quasi_exact_match_triviaqa",
    "quasi_exact_match_gsm8k",
    "quasi_exact_match_math",
    "pass@1:1_samples",
)


@dataclass
class EvalRunner:
    """Configure, execute, and parse a lighteval run."""

    model: str
    output_dir: str | Path
    task_group: str = "early-signal"
    num_processes: int = 8
    batch_size: int = 16
    max_samples: int | None = 1000
    main_process_port: int = 29600
    custom_tasks_path: str | Path = field(
        default_factory=lambda: REPO_ROOT / "src" / "incepedia" / "eval" / "lighteval_tasks.py"
    )
    lighteval_entry: str | Path | None = None
    model_args_extra: str = ""
    dry_run: bool = False

    # populated after run()
    returncode: int | None = None
    wall_seconds: float | None = None
    raw_results_json: Path | None = None
    scores: dict[str, float] = field(default_factory=dict)

    # ── command composition ───────────────────────────────────────────

    def resolve_lighteval_entry(self) -> Path:
        if self.lighteval_entry:
            return Path(self.lighteval_entry)
        candidates = [
            REPO_ROOT / "third_party" / "lighteval" / "run_evals_accelerate.py",
            REPO_ROOT / "third_party" / "lighteval" / "src" / "lighteval" / "__main__.py",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            "Could not locate lighteval entry script. Install it via "
            "scripts/bootstrap_env.sh or set EvalRunner(lighteval_entry=...)."
        )

    def build_command(self) -> list[str]:
        if self.task_group not in TASKS_GROUPS:
            raise KeyError(f"Unknown task_group '{self.task_group}'. Known: {list(TASKS_GROUPS)}")
        tasks_arg = TASKS_GROUPS[self.task_group]

        cmd = [
            "accelerate", "launch",
            f"--num_processes={self.num_processes}",
            f"--main_process_port={self.main_process_port}",
            str(self.resolve_lighteval_entry()),
            f"--model_args=pretrained={self.model}" + (f",{self.model_args_extra}" if self.model_args_extra else ""),
            "--custom_tasks", str(self.custom_tasks_path),
            "--output_dir", str(self.output_dir),
            "--override_batch_size", str(self.batch_size),
            "--tasks", tasks_arg,
        ]
        if self.max_samples is not None:
            cmd += ["--max_samples", str(self.max_samples)]
        return cmd

    # ── execution ─────────────────────────────────────────────────────

    def run(self) -> dict[str, float]:
        cmd = self.build_command()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        print("[eval] command:", " ".join(cmd), file=sys.stderr)
        if self.dry_run:
            self.returncode = 0
            return {}

        t0 = time.time()
        completed = subprocess.run(cmd, check=False)
        self.wall_seconds = time.time() - t0
        self.returncode = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(f"lighteval exited with code {completed.returncode}")

        self.scores = self._parse_results()
        return self.scores

    # ── result parsing ───────────────────────────────────────────────

    def _find_latest_results_json(self) -> Path:
        """lighteval writes results/<org>/<model>/results_<ts>.json ; pick newest."""
        results_root = Path(self.output_dir) / "results"
        if not results_root.exists():
            raise FileNotFoundError(f"No results directory under {results_root}")
        candidates = sorted(results_root.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No results_*.json under {results_root}")
        self.raw_results_json = candidates[-1]
        return self.raw_results_json

    def _parse_results(self) -> dict[str, float]:
        results_path = self._find_latest_results_json()
        with results_path.open() as f:
            payload = json.load(f)
        per_task: dict[str, dict[str, Any]] = payload.get("results", {}) or {}

        out: dict[str, float] = {}
        mmlu_cloze_scores: list[float] = []
        mmlu_mc_scores: list[float] = []

        for raw_task, metric_dict in per_task.items():
            if not isinstance(metric_dict, dict):
                continue
            # lighteval task keys look like: "custom|arc:easy|0" or "custom|mmlu_cloze:anatomy|0"
            m = re.match(r"^(?:custom\|)?([^|]+)(?:\|.*)?$", raw_task)
            task_core = m.group(1) if m else raw_task
            score = _pick_metric(metric_dict)
            if score is None:
                continue

            if task_core.startswith("mmlu_cloze:"):
                mmlu_cloze_scores.append(score)
            elif task_core.startswith("mmlu_mc:"):
                mmlu_mc_scores.append(score)
            else:
                col = _TASK_TO_COLUMN.get(task_core)
                if col is not None:
                    out[col] = float(score)

        if mmlu_cloze_scores:
            out["mmlu_cloze"] = float(sum(mmlu_cloze_scores) / len(mmlu_cloze_scores))
        if mmlu_mc_scores:
            out["mmlu_mc"] = float(sum(mmlu_mc_scores) / len(mmlu_mc_scores))

        return out

    # ── output ───────────────────────────────────────────────────────

    def write_metrics_json(self, path: str | Path, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "task_group": self.task_group,
            "num_processes": self.num_processes,
            "batch_size": self.batch_size,
            "max_samples": self.max_samples,
            "wall_seconds": self.wall_seconds,
            "raw_results_json": str(self.raw_results_json) if self.raw_results_json else None,
            **self.scores,
        }
        if extra:
            payload.update(extra)
        path.write_text(json.dumps(payload, indent=2))
        return path


def _pick_metric(metric_dict: dict[str, Any]) -> float | None:
    """Pick the canonical scalar score from a lighteval per-task metric dict."""
    for pref in _METRIC_PREFERENCE:
        if pref in metric_dict:
            val = metric_dict[pref]
            if isinstance(val, (int, float)):
                return float(val)
    # fall back: first numeric value
    for v in metric_dict.values():
        if isinstance(v, (int, float)):
            return float(v)
    return None
