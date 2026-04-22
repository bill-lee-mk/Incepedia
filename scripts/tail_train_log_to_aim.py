#!/usr/bin/env python
"""Sidecar: tail nanotron's train.log and push metrics into an Aim repo.

This is the TEMPORARY bridge from nanotron stdout → Aim UI (curve viewer),
pending the proper T2 fix where nanotron itself logs to Aim directly.

Design
------
- Reads the specified experiment's `train.log` in `tail -F` mode.
- Parses each `iteration: N / TOTAL | ...` line produced by
  `nanotron.trainer` (every `iteration_step_info_interval` steps).
- Pushes scalar metrics (lm_loss, lr, grad_norm, tokens_per_sec,
  tokens_per_sec_per_gpu, model_tflops_per_gpu, time_per_iteration_ms,
  consumed_tokens) into the Aim repo at `aim/` under a Run named after the
  experiment id (so you can compare multiple runs side-by-side).

- Safe: this is a READ-ONLY consumer of the log; it does NOT touch the
  training process.  If training stops, this sidecar just keeps polling.

Usage
-----
    python scripts/tail_train_log_to_aim.py exp_ref_cosmopedia_v2_qwen3_seed42
    # or with explicit paths:
    python scripts/tail_train_log_to_aim.py <exp_id> \
        --log-path experiments/<exp_id>/train.log \
        --aim-repo aim/
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

try:
    from aim import Run
    from aim.sdk.errors import MissingRunError
except ImportError:  # pragma: no cover
    sys.exit("[sidecar] `aim` package not installed in this env")

from incepedia.config import REPO_ROOT


# Strip ANSI color/style escapes that nanotron's coloured logger emits.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\[(?:1;3[0-9]|2;3|0)[0-9;]*m")

# An iteration line looks like:
#   04/21 10:23:57 [INFO|DP=0]: iteration: 131 / 22888 | consumed_tokens: 8.59M | ...
ITER_RE = re.compile(r"iteration:\s*(\d+)\s*/\s*(\d+)")
# `key: value` pair.  value may end with K / M / B / G / T or carry scientific
# notation.  `B` (billions) is what nanotron's human-format emits for
# `consumed_tokens >= 1e9` — we MUST recognise it or curves silently collapse
# to ~1 once the run crosses 1B tokens (visible on Aim as a vertical drop).
FIELD_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*):\s*([0-9eE+\-\.]+)\s*([KMBGT]?)")

# Names we don't want to push to Aim:
#  - iteration / global_batch_size: already encoded in the step counter or run const
#  - eta: nanotron emits it as "H:MM:SS" (e.g. "6:24:43") which our simple
#    numeric regex would misread as just the hour part; skip to avoid confusion
DROP = {"iteration", "global_batch_size", "eta"}

# Units → multiplier for suffix parsing.  Nanotron's human-format prints `B`
# for billions (10^9), distinct from giga `G` which it never uses.  We accept
# both to be safe.
UNIT_MULT = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}


def parse_line(line: str) -> tuple[int, int, dict[str, float]] | None:
    """Return (step, total_steps, {metric: value}) or None."""
    line = ANSI_RE.sub("", line)
    m = ITER_RE.search(line)
    if not m:
        return None
    step = int(m.group(1))
    total = int(m.group(2))

    metrics: dict[str, float] = {}
    # Everything after the iteration: X / Y tag is a "| k: v | k: v | ..." list.
    tail = line[m.end():]
    for part in tail.split("|"):
        part = part.strip()
        if not part:
            continue
        mm = FIELD_RE.match(part)
        if not mm:
            continue
        key, raw_val, unit = mm.group(1), mm.group(2), mm.group(3)
        if key in DROP:
            continue
        try:
            val = float(raw_val) * UNIT_MULT.get(unit, 1.0)
        except ValueError:
            continue
        metrics[key] = val
    return step, total, metrics


def follow(path: Path, poll_sec: float = 1.0):
    """Yield new lines appended to `path` (basic `tail -F`)."""
    last_inode = None
    f = None
    try:
        while True:
            if not path.exists():
                time.sleep(poll_sec)
                continue
            stat = path.stat()
            if f is None or stat.st_ino != last_inode:
                if f is not None:
                    f.close()
                f = path.open("r", encoding="utf-8", errors="replace")
                last_inode = stat.st_ino
                # Start at BOF on first open so we don't miss early iterations;
                # rotations start at BOF naturally (new file).
            line = f.readline()
            if not line:
                time.sleep(poll_sec)
                continue
            yield line
    finally:
        if f is not None:
            f.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_id", help="experiment id (e.g. exp_ref_cosmopedia_v2_qwen3_seed42)")
    p.add_argument("--log-path", type=Path, default=None,
                   help="override train.log path (default: experiments/<exp_id>/train.log)")
    p.add_argument("--aim-repo", type=Path, default=REPO_ROOT / "aim",
                   help=f"Aim repo dir (default: {REPO_ROOT/'aim'})")
    p.add_argument("--experiment-name", default="incepedia",
                   help="Aim 'experiment' group name used to cluster runs")
    p.add_argument("--poll-sec", type=float, default=1.0)
    args = p.parse_args()

    log_path = args.log_path or (REPO_ROOT / "experiments" / args.exp_id / "train.log")
    aim_repo = args.aim_repo
    aim_repo.mkdir(parents=True, exist_ok=True)

    # Aim's SDK refuses `run_hash=<str>` if the run doesn't exist yet, and
    # refuses to create one with a chosen hash.  So we cache the hash Aim gives
    # us on first creation and re-use it across restarts.
    hash_cache = aim_repo / f".sidecar_{args.exp_id}.hash"
    run = None
    if hash_cache.exists():
        cached = hash_cache.read_text().strip()
        try:
            run = Run(run_hash=cached, repo=str(aim_repo), experiment=args.experiment_name)
        except MissingRunError:
            run = None
    if run is None:
        run = Run(repo=str(aim_repo), experiment=args.experiment_name)
        hash_cache.write_text(run.hash)
    # Static tags for filtering in the UI.
    run["exp_id"] = args.exp_id
    run["source"] = "train.log sidecar"
    run["log_path"] = str(log_path)
    run.name = f"{args.exp_id} (log-sidecar)"

    print(f"[sidecar] exp_id={args.exp_id}")
    print(f"[sidecar] log    ={log_path}")
    print(f"[sidecar] aim    ={aim_repo}  (experiment={args.experiment_name})")
    print(f"[sidecar] run    ={run.hash}")
    print(f"[sidecar] open   =http://localhost:43800  → Runs tab → pick '{args.exp_id}-sidecar'")

    n_pushed = 0
    last_total = None
    for line in follow(log_path, poll_sec=args.poll_sec):
        parsed = parse_line(line)
        if parsed is None:
            continue
        step, total, metrics = parsed
        if total and last_total != total:
            run["train_steps_total"] = total
            last_total = total
        for k, v in metrics.items():
            run.track(v, name=k, step=step, context={"split": "train"})
        n_pushed += 1
        if n_pushed % 10 == 0:
            sample = ", ".join(f"{k}={metrics[k]:.3g}" for k in list(metrics)[:3])
            print(f"[sidecar] step={step}/{total}  pushed={n_pushed}  latest: {sample}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
