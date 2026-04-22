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

import yaml

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

_COMMON_ENV_PREAMBLE = '''\
import os
# datasets 4.x disables loading scripts by default; legacy benchmarks (piqa,
# siqa, boolq, hellaswag, etc.) need this flag to load.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
# Isolate our HF dataset cache from the shared system cache — the server is
# multi-tenant and system cache may contain `datasets` 4.x-format info files
# that our pinned 3.6 cannot parse (e.g. "Feature type 'List'" error).
os.environ.setdefault("HF_DATASETS_CACHE", {hf_datasets_cache!r})
os.environ.setdefault("HF_HOME", {hf_home!r})
# Force OFFLINE: datasets are pre-downloaded by scripts/prefetch_eval_datasets.py.
# Lighteval otherwise pings HF Hub `/api/datasets/.../tree/` for every task ×
# every rank, hitting the 500-req / 5-min IP rate limit (HTTP 429) and crashing
# the eval mid-launch (observed 2026-04-21 on the seed42 final eval).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
'''


_LAUNCHER_TEMPLATE = '''\
"""Auto-generated lighteval driver (HF accelerate backend). Do not edit."""
{preamble}
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


# Driver for nanotron-checkpoint evaluation.  Uses lighteval's native nanotron
# backend (NanotronLightevalModel) which loads sharded safetensors directly —
# no HF format conversion needed.  Distributed launch uses torchrun.
_NANOTRON_LAUNCHER_TEMPLATE = '''\
"""Auto-generated lighteval driver (nanotron backend). Do not edit."""
{preamble}
import yaml
from yaml import SafeLoader
from nanotron.config import (
    GeneralArgs, ModelArgs, TokenizerArgs, get_config_from_dict, get_config_from_file,
)
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.nanotron.nanotron_model import FullNanotronConfig, LightEvalConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

ckpt_yaml = {ckpt_yaml!r}
lighteval_yaml = {lighteval_yaml!r}

with open(ckpt_yaml) as f:
    nanotron_yaml = yaml.load(f, Loader=SafeLoader)

model_config, tokenizer_config, general_config = [
    get_config_from_dict(
        nanotron_yaml[key], config_class=cls,
        skip_unused_config_keys=True, skip_null_keys=True,
    )
    for key, cls in [("model", ModelArgs), ("tokenizer", TokenizerArgs), ("general", GeneralArgs)]
]
lighteval_config = get_config_from_file(lighteval_yaml, config_class=LightEvalConfig)
nanotron_config = FullNanotronConfig(lighteval_config, model_config, tokenizer_config, general_config)

evaluation_tracker = EvaluationTracker(
    output_dir=lighteval_config.logging.output_dir,
    save_details=lighteval_config.logging.save_details,
    nanotron_run_info=nanotron_config.nanotron_general,
)
pipeline_parameters = PipelineParameters(
    launcher_type=ParallelismManager.NANOTRON,
    job_id=os.environ.get("SLURM_JOB_ID", 0),
    nanotron_checkpoint_path=ckpt_yaml,
    dataset_loading_processes=lighteval_config.tasks.dataset_loading_processes,
    custom_tasks_directory=lighteval_config.tasks.custom_tasks,
    num_fewshot_seeds=1,
    max_samples=lighteval_config.tasks.max_samples,
)
pipeline = Pipeline(
    tasks=lighteval_config.tasks.tasks,
    pipeline_parameters=pipeline_parameters,
    evaluation_tracker=evaluation_tracker,
    model_config=nanotron_config,
)
pipeline.evaluate()
pipeline.show_results()
pipeline.save_and_push_results()
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

    def _is_nanotron_ckpt(self) -> bool:
        """Heuristic: a nanotron checkpoint dir contains both `config.yaml`
        and a `model/` subdir of sharded safetensors."""
        p = Path(self.model)
        return p.is_dir() and (p / "config.yaml").is_file() and (p / "model").is_dir()

    def _preamble(self, hf_cache_root: Path) -> str:
        return _COMMON_ENV_PREAMBLE.format(
            hf_datasets_cache=str(hf_cache_root / "datasets"),
            hf_home=str(hf_cache_root / "hf_home"),
        )

    def _write_lighteval_yaml(self, workdir: Path) -> Path:
        """Render a LightEvalConfig YAML for the nanotron backend."""
        cfg = {
            "logging": {
                "output_dir": str(self.output_dir),
                "save_details": False,
                "push_to_hub": False,
                "push_to_tensorboard": False,
                "public_run": False,
            },
            "tasks": {
                "tasks": self._tasks_arg(),
                "custom_tasks": str(self.custom_tasks_path),
                "max_samples": self.max_samples,
                "dataset_loading_processes": 1,   # serial per-rank to avoid 429
            },
            "parallelism": {
                "dp": self.num_processes,         # eval is dp-only
                "pp": 1,
                "tp": 1,
                "pp_engine": "1f1b",
                "tp_mode": "ALL_REDUCE",
                "tp_linear_async_communication": False,
                "expert_parallel_size": 1,
                "context_parallel_size": 1,
            },
            "batch_size": 0,
        }
        yaml_path = workdir / "lighteval_config.yaml"
        yaml_path.write_text(yaml.safe_dump(cfg))
        return yaml_path

    def _prefetch_tokenizer(self, hf_home: Path) -> None:
        """Pre-warm HF cache for the model's tokenizer so OFFLINE mode works.

        OFFLINE mode is required in the eval subprocess to avoid the
        lighteval-datasets 429 storm.  But OFFLINE also blocks the tokenizer
        download, so we must materialise the tokenizer files into the SAME
        isolated cache that the subprocess will read from
        (`<hf_home>/hub/models--<repo>/...`).

        We use `huggingface_hub.snapshot_download` with explicit `cache_dir=`
        because changing `HF_HOME` env var after `transformers` is already
        imported does NOT relocate the cache (it is captured at import time).
        """
        if not self._is_nanotron_ckpt():
            return
        try:
            ckpt_cfg = yaml.safe_load((Path(self.model) / "config.yaml").read_text())
            tok_name = (ckpt_cfg.get("tokenizer") or {}).get("tokenizer_name_or_path")
            if not tok_name:
                return
            cache_dir = hf_home / "hub"
            cache_dir.mkdir(parents=True, exist_ok=True)
            from huggingface_hub import snapshot_download
            print(f"[eval] pre-fetching tokenizer {tok_name} → {cache_dir}", file=sys.stderr)
            snapshot_download(
                repo_id=tok_name,
                cache_dir=str(cache_dir),
                allow_patterns=[
                    "tokenizer*",
                    "vocab*",
                    "merges.txt",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "config.json",
                    "generation_config.json",
                ],
            )
            print(f"[eval] tokenizer ready", file=sys.stderr)
        except Exception as e:
            print(f"[eval] tokenizer pre-fetch failed (non-fatal): {e!r}", file=sys.stderr)

    def _write_driver(self, workdir: Path) -> Path:
        from incepedia.config import DATA_DIR
        hf_cache_root = DATA_DIR / "hf_cache"
        hf_cache_root.mkdir(parents=True, exist_ok=True)
        # Pre-cache the tokenizer BEFORE any subprocess sets HF_HUB_OFFLINE=1.
        self._prefetch_tokenizer(hf_home=hf_cache_root / "hf_home")
        driver = workdir / "run_lighteval_driver.py"

        if self._is_nanotron_ckpt():
            ckpt_yaml = str(Path(self.model) / "config.yaml")
            lighteval_yaml = str(self._write_lighteval_yaml(workdir))
            driver.write_text(
                _NANOTRON_LAUNCHER_TEMPLATE.format(
                    preamble=self._preamble(hf_cache_root),
                    ckpt_yaml=ckpt_yaml,
                    lighteval_yaml=lighteval_yaml,
                )
            )
        else:
            driver.write_text(
                _LAUNCHER_TEMPLATE.format(
                    preamble=self._preamble(hf_cache_root),
                    model_args=self._model_args(),
                    tasks=self._tasks_arg(),
                    custom_tasks=str(self.custom_tasks_path),
                    output_dir=str(self.output_dir),
                    max_samples=self.max_samples,
                )
            )
        return driver

    def build_command(self, driver_path: Path) -> list[str]:
        if self._is_nanotron_ckpt():
            # Nanotron's distributed init expects torchrun, not accelerate.
            return [
                "torchrun",
                f"--nproc_per_node={self.num_processes}",
                f"--master_port={self.main_process_port}",
                str(driver_path),
            ]
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
            # lighteval 0.13 format is "task_name|num_fewshots" ; strip the fewshot suffix.
            if raw_task == "all":
                # Aggregate "all" row sometimes added by lighteval; we aggregate ourselves.
                continue
            task_core = raw_task.split("|")[0]
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
