#!/usr/bin/env python
"""Main experiment orchestrator.

Reads an `experiments/<exp_id>/config.yaml`, runs the full cycle:
    config → (tokenize if needed) → train → eval → metrics.json → INDEX → NAS sync

Usage:
    python scripts/run_experiment.py --config experiments/exp_ref_cosmopedia_v2/config.yaml
    python scripts/run_experiment.py --config <path> --eval-only
    python scripts/run_experiment.py --config <path> --dry-run

All steps respect the `track: 1|2` field in config for dispatching to the right
launcher (standalone vs cooldown-fork vs backbone-only).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from incepedia.config import REPO_ROOT
from incepedia.training.config import ExperimentConfig, Track, load_config


def _run(cmd: list[str], step: str) -> int:
    print(f"\n[orchestrator] === {step} ===")
    print(f"[orchestrator] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def run_pipeline(cfg: ExperimentConfig, *, eval_only: bool, dry_run: bool) -> int:
    exp_dir = cfg.exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sync config.yaml to NAS immediately (cheap insurance)
    rc = _run(
        ["bash", str(REPO_ROOT / "scripts" / "sync_to_nas.sh"), "config", cfg.exp_id],
        "sync config.yaml → NAS",
    )
    if rc != 0 and not dry_run:
        print(f"[warn] config sync returned {rc}; continuing", file=sys.stderr)

    # 2. Training (unless eval-only)
    if not eval_only:
        from incepedia.training.launcher import launch_training

        # Auto-spawn the train.log → Aim sidecar so Aim UI gets live curves
        # without modifying nanotron itself (TODO T2 — stopgap until a proper
        # nanotron → Aim hook lands in upstream).  The sidecar tails the same
        # train.log we already tee to disk, so it adds ~zero overhead.
        sidecar_proc = None
        if not dry_run:
            # Drop any stale hash cache so each training run gets its own Aim
            # Run (otherwise we'd append to the previous run's history).
            aim_repo = REPO_ROOT / "aim"
            stale_hash = aim_repo / f".sidecar_{cfg.exp_id}.hash"
            if stale_hash.exists():
                stale_hash.unlink()

            aim_log = exp_dir / "aim_sidecar.log"
            sidecar_cmd = [
                sys.executable, str(REPO_ROOT / "scripts" / "tail_train_log_to_aim.py"),
                cfg.exp_id,
            ]
            print(f"[orchestrator] === start Aim sidecar ===")
            print(f"[orchestrator] $ {' '.join(sidecar_cmd)} > {aim_log}")
            sidecar_proc = subprocess.Popen(
                sidecar_cmd,
                stdout=aim_log.open("ab"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(f"[orchestrator] sidecar pid={sidecar_proc.pid}")

        try:
            rc = launch_training(cfg, dry_run=dry_run)
        finally:
            if sidecar_proc is not None and sidecar_proc.poll() is None:
                sidecar_proc.terminate()
                try:
                    sidecar_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    sidecar_proc.kill()
                print(f"[orchestrator] sidecar stopped")
        if rc != 0 and not dry_run:
            print(f"[orchestrator] training failed with code {rc}", file=sys.stderr)
            return rc

        # Sync checkpoints (raw nanotron form — used for resume / cooldown-fork only)
        _run(
            ["bash", str(REPO_ROOT / "scripts" / "sync_to_nas.sh"), "ckpt", cfg.exp_id],
            "sync ckpt → NAS",
        )

    # 2.5  Convert latest nanotron ckpt to HF format
    # Eval / publishing always consume the HF copy (transformers / lighteval-
    # accelerate / vllm / lm-eval-harness all want the standard HF layout).
    # The lighteval-nanotron path in 0.13 is broken in many places (see ADR
    # 0009), so we always materialise an HF ckpt before eval.
    ckpt_dir = exp_dir / "ckpt"
    latest_ckpt = None
    if ckpt_dir.exists():
        ckpts = sorted(
            (p for p in ckpt_dir.iterdir() if p.is_dir() and (p / "config.yaml").is_file()),
            key=lambda p: int(p.name) if p.name.isdigit() else 0,
        )
        if ckpts:
            latest_ckpt = ckpts[-1]

    hf_ckpt_dir = exp_dir / "hf_ckpt"
    if not eval_only and latest_ckpt is not None and not dry_run:
        if hf_ckpt_dir.exists() and (hf_ckpt_dir / "model.safetensors").exists():
            print(f"[orchestrator] hf_ckpt already exists at {hf_ckpt_dir} — skip convert")
        else:
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "convert_nanotron_qwen2_to_hf.py"),
                    "--src", str(latest_ckpt),
                    "--dst", str(hf_ckpt_dir),
                ],
                f"convert nanotron→HF (step {latest_ckpt.name})",
            )
        _run(
            ["bash", str(REPO_ROOT / "scripts" / "sync_to_nas.sh"), "hf_ckpt", cfg.exp_id],
            "sync hf_ckpt → NAS",
        )
    elif eval_only and not hf_ckpt_dir.exists() and latest_ckpt is not None and not dry_run:
        # eval-only path: convert on demand if missing
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "convert_nanotron_qwen2_to_hf.py"),
                "--src", str(latest_ckpt),
                "--dst", str(hf_ckpt_dir),
            ],
            f"convert nanotron→HF (step {latest_ckpt.name})",
        )

    # 3. Evaluation
    from incepedia.eval.runner import EvalRunner

    # Eval reads the HF ckpt (or falls back to the raw nanotron dir if HF
    # conversion produced no output, which keeps `--eval-only` debug usable
    # against external HF model ids).
    if hf_ckpt_dir.exists() and (hf_ckpt_dir / "config.json").is_file():
        model_path = str(hf_ckpt_dir)
    else:
        model_path = str(latest_ckpt) if latest_ckpt else "REPLACE_WITH_MODEL_PATH"

    if dry_run or model_path == "REPLACE_WITH_MODEL_PATH":
        print(f"[orchestrator] eval model_path would be: {model_path}")
        if model_path == "REPLACE_WITH_MODEL_PATH":
            print("[orchestrator] no checkpoint found; skip eval")
            return 0

    runner = EvalRunner(
        model=model_path,
        output_dir=str(exp_dir / "eval"),
        task_group=cfg.eval.task_group,
        max_samples=cfg.eval.max_samples,
    )
    if not dry_run:
        scores = runner.run()
        runner.write_metrics_json(
            exp_dir / "metrics.json",
            extra={"track": int(cfg.track), "stage": cfg.stage.value, "exp_id": cfg.exp_id},
        )
        print(f"[orchestrator] eval done — {len(scores)} scores")

    # 4. Register in INDEX.parquet
    _run(
        ["python", str(REPO_ROOT / "scripts" / "index_experiment.py"), "add", cfg.exp_id],
        "INDEX.parquet update",
    )

    # 5. Final NAS sync
    _run(
        ["bash", str(REPO_ROOT / "scripts" / "sync_to_nas.sh"), "eval", cfg.exp_id],
        "sync eval → NAS",
    )

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="Path to experiment config.yaml")
    p.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate + register")
    p.add_argument("--dry-run", action="store_true", help="Print commands, don't execute")
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"[orchestrator] loaded config: {cfg.exp_id} (track={cfg.track}, stage={cfg.stage})")
    return run_pipeline(cfg, eval_only=args.eval_only, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
