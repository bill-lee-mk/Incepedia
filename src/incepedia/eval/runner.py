"""Lighteval 0.13 subprocess runner.

Wraps `accelerate launch` + `lighteval.main_accelerate.accelerate()` behind a Python
API, normalising per-task scores into metrics.json for INDEX.parquet.

Why we invoke Python programmatically instead of the `lighteval` CLI:
    lighteval 0.13's typer-based CLI is currently broken (typer/click version
    incompatibility). Calling `main_accelerate.accelerate()` from a tiny driver
    script inside `accelerate launch` sidesteps the CLI entirely and is future-
    proof if the CLI stabilises.

Usage
-----
    from incepedia.eval.runner import EvalRunner

    runner = EvalRunner(
        model="HuggingFaceTB/cosmo-1b",
        output_dir="experiments/exp_xxx/eval",
        task_group="early-signal",
        num_processes=8,
        max_samples=500,  # smoke test; set None or large for full runs
    )
    scores = runner.run()
    runner.write_metrics_json("experiments/exp_xxx/metrics.json")
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from incepedia.config import REPO_ROOT
from incepedia.eval.lighteval_tasks import TASKS_GROUPS

# Map lighteval task (core name) → column in INDEX.parquet.
# Keep in sync with scripts/index_experiment.py:SCHEMA_COLS.
# Tasks are prefixed with `incep_` in our port (to avoid collision with
# lighteval 0.13's built-in tasks of same name).
_TASK_TO_COLUMN: dict[str, str] = {
    "incep_hellaswag": "hellaswag",
    "incep_winogrande": "winogrande",
    "incep_piqa": "piqa",
    "incep_siqa": "siqa",
    "incep_openbookqa": "openbookqa",
    "incep_arc_easy": "arc_easy",
    "incep_arc_challenge": "arc_challenge",
    "incep_commonsense_qa": "commonsense_qa",
    "incep_boolq": "boolq",
    "incep_mmlu_pro_cloze": "mmlu_pro_cloze",
    "incep_mmlu_pro_mc": "mmlu_pro_mc",
    "incep_trivia_qa": "trivia_qa",
    "incep_gsm8k": "gsm8k_5shot",
}

# lighteval 0.13 exposes fewer per-metric variants; take the first numeric scalar
# preferring the normalised accuracy when present.
_METRIC_PREFERENCE = (
    "acc_norm_nospace",
    "acc_norm",
    "loglikelihood_acc",
    "acc",
    "exact_match",
    "em",
    "pass@1:1_samples",
    "pass@1",
)

_LAUNCHER_TEMPLATE = '''\
"""Auto-generated lighteval driver. Do not edit.

Written by incepedia.eval.runner.EvalRunner. Invoked via `accelerate launch`.
"""
import os
# datasets 4.x disables loading scripts by default; legacy benchmarks (piqa,
# siqa, boolq, hellaswag, etc.) need this flag to load.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")

from lighteval.main_accelerate import accelerate

accelerate(
    model_args={model_args!r},
    tasks={tasks!r},
    custom_tasks={custom_tasks!r},
    output_dir={output_dir!r},
    max_samples={max_samples!r},
    save_details=False,
)
'''


@dataclass
class EvalRunner:
    """Execute lighteval against a model and normalise scores."""

    model: str
    output_dir: str | Path
    task_group: str = "early-signal"
    num_processes: int = 8
    main_process_port: int = 29600
    max_samples: int | None = 500
    custom_tasks_path: str | Path = field(
        default_factory=lambda: REPO_ROOT / "src" / "incepedia" / "eval" / "lighteval_tasks.py"
    )
    model_args_extra: str = ""  # appended after `pretrained=<model>`, comma-separated kwargs
    mixed_precision: str = "bf16"  # "no"|"fp16"|"bf16"
    dry_run: bool = False

    # populated after run()
    returncode: int | None = None
    wall_seconds: float | None = None
    raw_results_json: Path | None = None
    scores: dict[str, float] = field(default_factory=dict)

    # ── command composition ───────────────────────────────────────────

    def _model_args(self) -> str:
        # lighteval 0.13 renamed `pretrained` -> `model_name` in TransformersModelConfig.
        base = f"model_name={self.model}"
        if self.model_args_extra:
            base = f"{base},{self.model_args_extra}"
        return base

    def _tasks_arg(self) -> str:
        if self.task_group not in TASKS_GROUPS:
            raise KeyError(f"Unknown task_group '{self.task_group}'. Known: {list(TASKS_GROUPS)}")
        return TASKS_GROUPS[self.task_group]

    def _write_driver(self, workdir: Path) -> Path:
        driver = workdir / "run_lighteval_driver.py"
        driver.write_text(
            _LAUNCHER_TEMPLATE.format(
                model_args=self._model_args(),
                tasks=self._tasks_arg(),
                custom_tasks=str(self.custom_tasks_path),
                output_dir=str(self.output_dir),
                max_samples=self.max_samples,
            )
        )
        return driver

    def build_command(self, driver_path: Path) -> list[str]:
        return [
            "accelerate", "launch",
            f"--num_processes={self.num_processes}",
            f"--main_process_port={self.main_process_port}",
            f"--mixed_precision={self.mixed_precision}",
            str(driver_path),
        ]

    # ── execution ─────────────────────────────────────────────────────

    def run(self) -> dict[str, float]:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="incepedia_eval_"))
        try:
            driver = self._write_driver(tmp)
            cmd = self.build_command(driver)
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
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ── result parsing ───────────────────────────────────────────────

    def _find_latest_results_json(self) -> Path:
        """lighteval 0.13 writes results/<org>/<model>/results_<ts>.json ; pick newest."""
        out = Path(self.output_dir)
        candidates = sorted(out.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No results_*.json under {out}")
        self.raw_results_json = candidates[-1]
        return self.raw_results_json

    def _parse_results(self) -> dict[str, float]:
        payload = json.loads(self._find_latest_results_json().read_text())
        per_task: dict[str, dict[str, Any]] = payload.get("results", {}) or {}

        out: dict[str, float] = {}
        mmlu_cloze_scores: list[float] = []
        mmlu_mc_scores: list[float] = []

        for raw_task, metric_dict in per_task.items():
            if not isinstance(metric_dict, dict):
                continue
            # raw_task examples: "incep_arc_easy|0", "incep_mmlu_cloze:anatomy|0", "all"
            # lighteval 0.13 strips the `suite|` prefix; some versions still emit it.
            if raw_task == "all":
                # Aggregate "all" row sometimes added by lighteval; we aggregate ourselves.
                continue
            m = re.match(r"^(?:[^|]+\|)?([^|]+)(?:\|.*)?$", raw_task)
            task_core = m.group(1) if m else raw_task
            score = _pick_metric(metric_dict)
            if score is None:
                continue

            if task_core.startswith("incep_mmlu_cloze:"):
                mmlu_cloze_scores.append(score)
            elif task_core.startswith("incep_mmlu_mc:"):
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
    for v in metric_dict.values():
        if isinstance(v, (int, float)):
            return float(v)
    return None
