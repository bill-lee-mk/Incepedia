#!/usr/bin/env python3
"""scripts/train_watchdog.py — health check + auto-resume for detached training.

Designed to be called every 5 minutes by cron. Detects 4 failure modes and,
on detection, auto-relaunches via run_detached.sh (which re-uses
nanotron's auto-resume from the latest valid ckpt).

Failure modes detected:
  1. PID file exists but process is dead       → relaunch
  2. PID alive but train.log not modified >30m → kill + relaunch (zombie/stuck)
  3. Iteration counter not advancing in 3 polls → kill + relaunch
  4. PID dead AND no PID file at all (clean exit)
       → if metrics.json present  → no action (training finished cleanly)
       → if metrics.json absent    → relaunch (probably crashed and not yet retried)

Watchdog state stored in: logs/detached/watchdog_<name>.state.json
   { last_poll_ts, last_iter, restart_count, alerts: [...] }

USAGE:
    python scripts/train_watchdog.py --name <name> --resume-cmd '<cmd>'

CRON:
    */5 * * * * cd /home/ubuntu/lilei/projects/Incepedia && \\
        python scripts/train_watchdog.py --config-file logs/detached/watchdog.yaml \\
        >> logs/detached/watchdog.log 2>&1

The --config-file mode allows watching multiple training jobs from one cron entry.
Format: list of {name, resume_cmd, log_path, metrics_path} dicts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DETACHED_DIR = REPO_ROOT / "logs" / "detached"
DETACHED_DIR.mkdir(parents=True, exist_ok=True)

ITERATION_RE = re.compile(r"iteration:\s*(\d+)\s*/\s*(\d+)")


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we can't signal


def _read_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"last_iter": -1, "stagnant_polls": 0, "restart_count": 0, "alerts": []}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {"last_iter": -1, "stagnant_polls": 0, "restart_count": 0, "alerts": []}


def _write_state(state_path: Path, state: dict) -> None:
    state["last_poll_ts"] = _ts()
    state_path.write_text(json.dumps(state, indent=2))


def _latest_iter_from_log(log_path: Path, max_tail_bytes: int = 1_000_000) -> int:
    """Tail the log and return the highest iteration number found, else -1."""
    if not log_path.exists():
        return -1
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            if size > max_tail_bytes:
                f.seek(-max_tail_bytes, os.SEEK_END)
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return -1
    matches = ITERATION_RE.findall(tail)
    if not matches:
        return -1
    return max(int(m[0]) for m in matches)


def _file_age_seconds(p: Path) -> float | None:
    if not p.exists():
        return None
    return time.time() - p.stat().st_mtime


def _gpu_used() -> bool:
    """True iff nvidia-smi reports any GPU >50% util."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        utils = [int(line.strip()) for line in out.splitlines() if line.strip()]
        return any(u >= 50 for u in utils)
    except Exception:
        return False


def _alert(state: dict, msg: str) -> None:
    state["alerts"].append({"ts": _ts(), "msg": msg})
    state["alerts"] = state["alerts"][-50:]
    _log(f"ALERT: {msg}")


def _relaunch(name: str, resume_cmd: list[str], state: dict) -> None:
    """Call run_detached.sh with the original training command."""
    state["restart_count"] = state.get("restart_count", 0) + 1
    _alert(state, f"auto-relaunch attempt #{state['restart_count']}")
    cmd = [str(REPO_ROOT / "scripts" / "run_detached.sh"), name, "--", *resume_cmd]
    _log(f"relaunching: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
        _log(f"relaunch rc={r.returncode}")
        if r.stdout:
            _log(f"stdout:\n{r.stdout}")
        if r.stderr:
            _log(f"stderr:\n{r.stderr}")
    except Exception as e:
        _alert(state, f"relaunch FAILED: {e!r}")


def check_one(
    *,
    name: str,
    resume_cmd: list[str],
    log_path: Path,
    metrics_path: Path | None,
    pid_file: Path,
    state_path: Path,
    stale_log_seconds: float,
    max_stagnant_polls: int,
    max_restarts: int,
) -> None:
    state = _read_state(state_path)
    if state.get("restart_count", 0) >= max_restarts:
        _log(f"[{name}] max restarts ({max_restarts}) reached; manual intervention required")
        return

    pid = None
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except Exception:
            pid = None

    pid_alive = pid is not None and _alive(pid)

    # Mode 4: clean finish — pid gone AND metrics.json present
    if not pid_alive and metrics_path and metrics_path.exists():
        _log(f"[{name}] training finished cleanly (metrics.json present); no action")
        _write_state(state_path, state)
        return

    log_age = _file_age_seconds(log_path)
    latest_iter = _latest_iter_from_log(log_path)

    # Mode 1: process dead, training not finished → relaunch
    if not pid_alive:
        _alert(state, f"PID {pid} dead AND no metrics.json; will relaunch")
        _relaunch(name, resume_cmd, state)
        _write_state(state_path, state)
        return

    # Mode 2: process alive but log stale (>30 min) → kill + relaunch
    if log_age is not None and log_age > stale_log_seconds:
        _alert(state, f"log stale: {log_age:.0f}s old (>{stale_log_seconds}s); killing PID {pid}")
        try:
            os.kill(pid, 9)
        except Exception as e:
            _alert(state, f"kill failed: {e!r}")
        time.sleep(3)
        _relaunch(name, resume_cmd, state)
        _write_state(state_path, state)
        return

    # Mode 3: iteration counter not advancing across N polls → kill + relaunch
    if latest_iter <= state.get("last_iter", -1):
        state["stagnant_polls"] = state.get("stagnant_polls", 0) + 1
        if state["stagnant_polls"] >= max_stagnant_polls:
            _alert(state, f"iteration {latest_iter} unchanged across {max_stagnant_polls} polls; killing PID {pid}")
            try:
                os.kill(pid, 9)
            except Exception as e:
                _alert(state, f"kill failed: {e!r}")
            time.sleep(3)
            state["stagnant_polls"] = 0
            _relaunch(name, resume_cmd, state)
            _write_state(state_path, state)
            return
    else:
        state["stagnant_polls"] = 0

    # All healthy
    _log(f"[{name}] healthy: PID {pid} alive, log {log_age:.0f}s old, "
         f"iter {latest_iter} (was {state.get('last_iter', -1)}), "
         f"restarts {state.get('restart_count', 0)}, GPU active={_gpu_used()}")
    state["last_iter"] = latest_iter
    _write_state(state_path, state)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", help="job name (matches logs/detached/<name>.{log,pid})")
    p.add_argument("--resume-cmd", help="JSON list of cmd argv (e.g. '[\"python\", \"x.py\"]')")
    p.add_argument("--log-path", help="override default logs/detached/<name>.log")
    p.add_argument("--metrics-path", help="if this file exists and PID dead, treat as clean finish")
    p.add_argument("--config-file", help="multi-job mode: YAML/JSON list of jobs to monitor")
    p.add_argument("--stale-log-seconds", type=int, default=1800)  # 30 min
    p.add_argument("--max-stagnant-polls", type=int, default=3)    # 3 × 5min = 15 min stuck
    p.add_argument("--max-restarts", type=int, default=10)
    args = p.parse_args()

    if args.config_file:
        cfg_path = Path(args.config_file)
        if not cfg_path.exists():
            _log(f"config file not found: {cfg_path}")
            return 1
        if cfg_path.suffix in (".yaml", ".yml"):
            import yaml
            jobs = yaml.safe_load(cfg_path.read_text())
        else:
            jobs = json.loads(cfg_path.read_text())
    else:
        if not args.name or not args.resume_cmd:
            _log("--name and --resume-cmd required (or use --config-file)")
            return 2
        jobs = [{
            "name": args.name,
            "resume_cmd": json.loads(args.resume_cmd),
            "log_path": args.log_path,
            "metrics_path": args.metrics_path,
        }]

    for job in jobs:
        name = job["name"]
        resume_cmd = job["resume_cmd"]
        log_path = Path(job.get("log_path") or DETACHED_DIR / f"{name}.log")
        metrics_path = Path(job["metrics_path"]) if job.get("metrics_path") else None
        pid_file = Path(job.get("pid_file") or DETACHED_DIR / f"{name}.pid")
        state_path = DETACHED_DIR / f"watchdog_{name}.state.json"
        check_one(
            name=name, resume_cmd=resume_cmd, log_path=log_path,
            metrics_path=metrics_path, pid_file=pid_file, state_path=state_path,
            stale_log_seconds=args.stale_log_seconds,
            max_stagnant_polls=args.max_stagnant_polls,
            max_restarts=args.max_restarts,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
