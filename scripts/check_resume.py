#!/usr/bin/env python
"""Preview what `scripts/run_experiment.py <cfg>` would do for resume.

Prints:
  - whether training will START FRESH or AUTO-RESUME
  - which ckpt step would be the resume point (if any)
  - what total + remaining train steps look like
  - any stale / incomplete ckpts found

Non-destructive; runs the launcher's dry-run machinery and reports.

Usage:
    python scripts/check_resume.py experiments/<exp_id>/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def inspect(cfg_path: Path) -> int:
    from incepedia.training.config import load_config
    from incepedia.training.launcher import build_nanotron_yaml

    cfg = load_config(cfg_path)
    exp_dir = cfg.exp_dir
    ckpt_root = exp_dir / "ckpt"

    print(f"── resume check for {cfg.exp_id} ─────────────────────────")
    print(f"  exp_dir     : {exp_dir}")
    print(f"  ckpt root   : {ckpt_root}  ({'exists' if ckpt_root.is_dir() else 'missing'})")

    if ckpt_root.is_dir():
        all_ckpts = sorted(
            (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        valid, broken = [], []
        for p in all_ckpts:
            ok = ((p / "config.yaml").is_file()
                  and (p / "model").is_dir()
                  and (p / "checkpoint_metadata.json").is_file())
            (valid if ok else broken).append(p)
        print(f"  checkpoints : {len(all_ckpts)} total, {len(valid)} valid, {len(broken)} broken")
        for p in valid[-5:]:
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 3)
            print(f"    ✓ {p.name:>6s}   {sz:.1f} GB")
        for p in broken:
            print(f"    ✗ {p.name:>6s}   (missing config/model/metadata)")
    else:
        valid = []

    print()
    # Build the yaml to see resume_path decision + step math (prints to stderr).
    print("── launcher yaml decision ────────────────────────────────")
    y = build_nanotron_yaml(cfg)
    resume = y["checkpoints"]["resume_checkpoint_path"]
    train_steps = y["tokens"]["train_steps"]
    grad_accum = y["tokens"]["batch_accumulation_per_replica"]
    print(f"  resume_checkpoint_path : {resume or 'None (fresh start)'}")
    print(f"  train_steps (total)    : {train_steps:,}")
    print(f"  grad_accum_per_replica : {grad_accum}")

    if resume and valid:
        # nanotron resumes at stored step + 1; infer from dirname
        step = int(Path(resume).name)
        remaining = train_steps - step
        pct = step / train_steps * 100
        print(f"  → would RESUME from step {step:,} ({pct:.1f}% done)")
        print(f"  → remaining            : {remaining:,} steps")
    else:
        print(f"  → would START FRESH at step 1")
        print(f"  → total to train       : {train_steps:,} steps")
    print()
    print("To actually launch training:  python scripts/run_experiment.py --config", cfg_path)
    print("To force a clean restart  :  INCEPEDIA_FRESH_START=1 python scripts/run_experiment.py --config", cfg_path)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, help="experiment config.yaml")
    args = p.parse_args()
    return inspect(args.config)


if __name__ == "__main__":
    sys.exit(main())
