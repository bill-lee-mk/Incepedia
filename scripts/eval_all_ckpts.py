#!/usr/bin/env python
"""Evaluate every intermediate checkpoint of an experiment.

Used to produce a TRAINING PROGRESSION CURVE (e.g. for FinePhrase Figure 1
overlay) — converts each `experiments/<exp>/ckpt/<step>/` to HF format,
runs lighteval (configurable task_group), writes per-step metrics to
`experiments/<exp>/eval_curve/step_<NNNNN>/metrics.json`, and emits a
combined `eval_curve/curve.json` for plotting.

Skips:
    - already-evaluated steps (resumes safely)
    - the latest ckpt if `--skip-final` (orchestrator already evals it)
    - non-numeric ckpt dirs

Usage:
    python scripts/eval_all_ckpts.py --config experiments/<exp>/config.yaml
    python scripts/eval_all_ckpts.py --config <path> --task-group finephrase-12
    python scripts/eval_all_ckpts.py --config <path> --every 2000   # eval every 2k steps
    python scripts/eval_all_ckpts.py --config <path> --max-samples 500  # quick smoke
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from incepedia.config import REPO_ROOT
from incepedia.training.config import load_config


def _list_ckpts(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, path), ...] sorted by step ascending."""
    if not ckpt_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        if not (p.is_dir() and p.name.isdigit()):
            continue
        # require a "valid" ckpt: model/ + checkpoint_metadata.json + config.yaml
        if not (
            (p / "model").is_dir()
            and (p / "checkpoint_metadata.json").is_file()
            and (p / "config.yaml").is_file()
        ):
            continue
        out.append((int(p.name), p))
    out.sort(key=lambda x: x[0])
    return out


def _convert_to_hf(nanotron_ckpt: Path, hf_dst: Path) -> int:
    if hf_dst.exists() and (hf_dst / "model.safetensors").is_file():
        print(f"[eval-curve] HF ckpt already at {hf_dst}; skip convert", flush=True)
        return 0
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "convert_nanotron_qwen2_to_hf.py"),
        "--src", str(nanotron_ckpt),
        "--dst", str(hf_dst),
    ]
    print(f"[eval-curve] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def _eval_one(
    hf_ckpt: Path,
    out_dir: Path,
    task_group: str,
    max_samples: int | None,
    num_processes: int,
) -> dict[str, float]:
    """Run lighteval on one HF ckpt; returns flat scores dict."""
    from incepedia.eval.runner import EvalRunner
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = EvalRunner(
        model=str(hf_ckpt),
        output_dir=str(out_dir),
        task_group=task_group,
        num_processes=num_processes,
        max_samples=max_samples,
    )
    scores = runner.run()
    runner.write_metrics_json(out_dir / "metrics.json")
    return scores


def _curve_macro(scores: dict[str, float]) -> float:
    """Macro mean across all reported scores. Filters out non-numeric."""
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--task-group", default=None, help="override cfg.eval.task_group")
    p.add_argument("--every", type=int, default=None,
                   help="only eval ckpts whose step is a multiple of this (default: all valid)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="lighteval max_samples cap for fast smoke; default = full")
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--skip-final", action="store_true",
                   help="skip the highest-step ckpt (orchestrator already evals it)")
    p.add_argument("--keep-hf", action="store_true",
                   help="keep per-step HF dir after eval (default: delete to save disk)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    exp_dir = cfg.exp_dir
    ckpt_dir = exp_dir / "ckpt"
    eval_curve_dir = exp_dir / "eval_curve"
    eval_curve_dir.mkdir(parents=True, exist_ok=True)
    task_group = args.task_group or cfg.eval.task_group
    max_samples = args.max_samples if args.max_samples is not None else cfg.eval.max_samples

    ckpts = _list_ckpts(ckpt_dir)
    if not ckpts:
        print(f"[eval-curve] no valid ckpts under {ckpt_dir}", file=sys.stderr)
        return 1

    if args.every:
        ckpts = [(s, p) for (s, p) in ckpts if s % args.every == 0]
    if args.skip_final and ckpts:
        ckpts = ckpts[:-1]

    print(f"[eval-curve] exp={cfg.exp_id}  task_group={task_group}  max_samples={max_samples}", flush=True)
    print(f"[eval-curve] {len(ckpts)} ckpts to eval: steps={[s for (s, _) in ckpts]}", flush=True)
    if args.dry_run:
        return 0

    curve_rows: list[dict] = []
    t0 = time.time()
    for i, (step, nano_ckpt) in enumerate(ckpts, 1):
        step_str = f"{step:07d}"
        out_dir = eval_curve_dir / f"step_{step_str}"
        metrics_json = out_dir / "metrics.json"
        if metrics_json.exists():
            try:
                payload = json.loads(metrics_json.read_text())
                scores = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
                print(f"[eval-curve] [{i}/{len(ckpts)}] step {step}: cached "
                      f"({len(scores)} scores, macro={_curve_macro(scores):.4f})", flush=True)
                curve_rows.append({"step": step, "macro": _curve_macro(scores), **scores})
                continue
            except Exception as e:
                print(f"[eval-curve] cache read failed at step {step}: {e}; re-evaluating", flush=True)

        # Convert nanotron → HF (in temp HF subdir)
        hf_dst = out_dir / "hf_ckpt"
        rc = _convert_to_hf(nano_ckpt, hf_dst)
        if rc != 0:
            print(f"[eval-curve] convert failed for step {step} (rc={rc}); skip", file=sys.stderr)
            continue

        ts = time.time()
        try:
            scores = _eval_one(hf_dst, out_dir, task_group, max_samples, args.num_processes)
        except Exception as e:
            print(f"[eval-curve] eval failed for step {step}: {e}", file=sys.stderr)
            continue
        dt = time.time() - ts
        macro = _curve_macro(scores)
        print(f"[eval-curve] [{i}/{len(ckpts)}] step {step}: {len(scores)} scores, "
              f"macro={macro:.4f}, took {dt/60:.1f} min", flush=True)
        curve_rows.append({"step": step, "macro": macro, **scores})

        if not args.keep_hf and hf_dst.exists():
            shutil.rmtree(hf_dst, ignore_errors=True)

    # Combine into single curve.json (for plot script consumption)
    curve_path = eval_curve_dir / "curve.json"
    curve_path.write_text(json.dumps({
        "exp_id": cfg.exp_id,
        "task_group": task_group,
        "global_batch_tokens": cfg.training.global_batch_tokens,
        "rows": sorted(curve_rows, key=lambda r: r["step"]),
    }, indent=2))
    total_min = (time.time() - t0) / 60
    print(f"[eval-curve] DONE — wrote {len(curve_rows)} rows to {curve_path} (total {total_min:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
