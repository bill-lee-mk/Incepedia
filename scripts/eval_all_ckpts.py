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

Parallelism:
    On a node with multiple GPUs, single-GPU evals of a 1.7B model are typically
    bottlenecked by per-task setup, not compute, so running N evals in parallel
    (each on total_gpus/N cards) is faster than sequential 8-card DP. Use
    --parallel-jobs N --gpus-per-job K to spawn N concurrent workers; or
    --probe-strategies '1x8,2x4,4x2,8x1' to benchmark the FIRST ckpt under each
    strategy and pick the fastest for the remaining ckpts. Probing reuses the
    eval results so no work is wasted (cheapest probe winner = first eval).

Usage:
    # sequential, full DP (default, slowest but safest):
    python scripts/eval_all_ckpts.py --config experiments/<exp>/config.yaml

    # 8-way parallel, single-GPU per ckpt (fastest for tiny models):
    python scripts/eval_all_ckpts.py --config <path> --parallel-jobs 8 --gpus-per-job 1

    # adaptive: probe 4 strategies on first ckpt, then run rest with the winner:
    python scripts/eval_all_ckpts.py --config <path> --probe-strategies 1x8,2x4,4x2,8x1

    # subsetting / smoke:
    python scripts/eval_all_ckpts.py --config <path> --every 2000        # skip every other
    python scripts/eval_all_ckpts.py --config <path> --max-samples 500   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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
    """Convert nanotron ckpt → HF safetensors. Idempotent + cheap (CPU + 1 GPU briefly)."""
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


@dataclass
class EvalJob:
    """One ckpt evaluation that needs to be dispatched to a GPU group."""

    step: int
    nanotron_ckpt: Path
    hf_dst: Path
    out_dir: Path
    task_group: str
    max_samples: int | None
    keep_hf: bool


@dataclass
class GpuGroup:
    """A subset of GPUs assigned to one parallel worker."""

    worker_id: int
    gpu_ids: list[int]
    master_port: int


def _build_worker_script(repo_root: Path) -> Path:
    """Write a small driver script the workers exec; lives in a tmp file."""
    return repo_root / "scripts" / "_eval_one_ckpt_worker.py"


def _run_eval_subprocess(job: EvalJob, group: GpuGroup) -> tuple[int, float, int]:
    """Spawn a child python that evaluates one ckpt on a fixed GPU subset.

    Returns (step, wall_seconds, returncode).
    """
    worker = _build_worker_script(REPO_ROOT)
    cmd = [
        sys.executable, str(worker),
        "--hf-ckpt", str(job.hf_dst),
        "--out-dir", str(job.out_dir),
        "--task-group", job.task_group,
        "--num-processes", str(len(group.gpu_ids)),
        "--master-port", str(group.master_port),
    ]
    if job.max_samples is not None:
        cmd += ["--max-samples", str(job.max_samples)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in group.gpu_ids)

    log_path = job.out_dir / "eval_worker.log"
    job.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[eval-curve] launch worker w{group.worker_id} step={job.step} "
        f"GPUs={group.gpu_ids} port={group.master_port} → {log_path.name}",
        flush=True,
    )
    t0 = time.time()
    with open(log_path, "ab") as logfh:
        completed = subprocess.run(cmd, env=env, stdout=logfh, stderr=subprocess.STDOUT)
    return job.step, time.time() - t0, completed.returncode


def _curve_macro(scores: dict[str, float]) -> float:
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _read_metrics(out_dir: Path) -> dict[str, float] | None:
    p = out_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
        return {k: v for k, v in payload.items() if isinstance(v, (int, float))}
    except Exception:
        return None


def _gpu_groups(parallel_jobs: int, gpus_per_job: int, base_port: int = 29600) -> list[GpuGroup]:
    """Split [0..N) GPUs into `parallel_jobs` groups of size `gpus_per_job`."""
    out: list[GpuGroup] = []
    for w in range(parallel_jobs):
        gpu_ids = list(range(w * gpus_per_job, (w + 1) * gpus_per_job))
        out.append(GpuGroup(worker_id=w, gpu_ids=gpu_ids, master_port=base_port + 100 * w))
    return out


def _parse_strategy(s: str) -> tuple[int, int]:
    """`'4x2'` → (parallel_jobs=4, gpus_per_job=2)."""
    n_str, k_str = s.lower().split("x")
    return int(n_str), int(k_str)


def _detect_gpu_count() -> int:
    try:
        import torch  # local import; eval host has it
        return torch.cuda.device_count()
    except Exception:
        return 8  # safe default for our hosts


def _dispatch_pool(
    jobs: list[EvalJob],
    groups: list[GpuGroup],
) -> dict[int, tuple[float, int]]:
    """Run `jobs` across GPU groups, len(groups) at a time. Returns {step: (wall, rc)}."""
    import concurrent.futures as cf

    results: dict[int, tuple[float, int]] = {}
    pending: list[EvalJob] = list(jobs)
    free: list[GpuGroup] = list(groups)

    futs: dict[cf.Future, GpuGroup] = {}
    with cf.ThreadPoolExecutor(max_workers=len(groups)) as ex:
        while pending or futs:
            # launch as many as we have free groups
            while pending and free:
                job = pending.pop(0)
                grp = free.pop(0)
                fut = ex.submit(_run_eval_subprocess, job, grp)
                futs[fut] = grp
            # wait for any to finish
            done, _ = cf.wait(list(futs.keys()), return_when=cf.FIRST_COMPLETED)
            for f in done:
                grp = futs.pop(f)
                step, wall, rc = f.result()
                results[step] = (wall, rc)
                free.append(grp)
                print(
                    f"[eval-curve] worker w{grp.worker_id} step={step} done in {wall/60:.1f}min "
                    f"(rc={rc})",
                    flush=True,
                )
    return results


def _do_probe(
    probe_ckpt: tuple[int, Path],
    eval_curve_dir: Path,
    task_group: str,
    max_samples: int | None,
    strategies: list[tuple[int, int]],
    keep_hf: bool,
    total_gpus: int,
) -> tuple[int, int]:
    """Run the same ckpt under each strategy, pick fastest. Returns (parallel_jobs, gpus_per_job).

    The probe reuses metrics.json across runs by writing each strategy's output
    to a sibling subdir (`step_<NNNNN>__probeNxK/`) so we have evidence preserved.
    The WINNING strategy's run becomes the canonical step_<NNNNN>/metrics.json
    (we copy it). This way we never waste eval work."""
    step, nano_ckpt = probe_ckpt
    step_str = f"{step:07d}"
    print(f"\n[eval-curve][probe] ⏱ benchmarking {len(strategies)} strategies on step {step}",
          flush=True)

    timings: dict[tuple[int, int], float] = {}

    for n, k in strategies:
        if n * k > total_gpus:
            print(f"[eval-curve][probe] skip {n}x{k}: needs {n*k} GPUs, have {total_gpus}",
                  flush=True)
            continue
        probe_dir = eval_curve_dir / f"step_{step_str}__probe{n}x{k}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        # Each probe needs its own HF ckpt to be safe (parallel write of HF ckpt
        # is fine since destinations differ; keep a single canonical HF copy).
        hf_dst = eval_curve_dir / f"step_{step_str}" / "hf_ckpt"
        rc = _convert_to_hf(nano_ckpt, hf_dst)
        if rc != 0:
            raise RuntimeError(f"HF convert failed for probe step {step}, rc={rc}")

        # remove any old metrics from a previous probe attempt at same N×K
        for stale in probe_dir.glob("metrics.json"):
            stale.unlink()

        job = EvalJob(
            step=step, nanotron_ckpt=nano_ckpt, hf_dst=hf_dst,
            out_dir=probe_dir, task_group=task_group, max_samples=max_samples,
            keep_hf=True,  # never delete HF during probe
        )
        groups = _gpu_groups(parallel_jobs=n, gpus_per_job=k)
        # one ckpt × n groups = run n parallel BUT we only have 1 ckpt to eval here.
        # Probe semantics: 1 ckpt at strategy n×k means: use n=1 worker with k GPUs
        # (we want to time ONE ckpt under that group size).
        results = _dispatch_pool([job], [groups[0]])
        wall, rc = results[step]
        if rc != 0:
            print(f"[eval-curve][probe] {n}x{k} FAILED rc={rc}; see {probe_dir}/eval_worker.log",
                  flush=True)
            timings[(n, k)] = float("inf")
        else:
            print(f"[eval-curve][probe] {n}x{k} (per-ckpt {len(_gpu_groups(n,k)[0].gpu_ids)}-GPU) "
                  f"= {wall/60:.2f} min/ckpt", flush=True)
            timings[(n, k)] = wall

    # Choose strategy: minimise TOTAL wall to finish all remaining ckpts.
    # Per-ckpt wall depends only on k (gpus_per_job). With n parallel workers
    # and C remaining ckpts (= 21 - 1 probed = 20 typically), total wall ≈
    # ceil(C/n) × per_ckpt_wall(k). We compute and pick the minimum.
    # NOTE: the probe also gave us 1 finished eval, so 1 fewer ckpt remains.
    remaining_ckpts_count = max(1, 20)  # caller has total len; we approx for ranking
    summary = []
    best_total = float("inf")
    best = list(timings.keys())[0]
    for (n, k), wall in timings.items():
        # waves needed
        waves = -(-remaining_ckpts_count // n)  # ceil div
        total = waves * wall
        summary.append((f"{n}x{k}", f"{wall/60:.2f}min/ckpt", f"≈{total/60:.1f}h total"))
        if total < best_total:
            best_total = total
            best = (n, k)

    print(f"\n[eval-curve][probe] strategy summary: {summary}", flush=True)
    print(f"[eval-curve][probe] WINNER (min total wall): {best[0]}x{best[1]} "
          f"@ {timings[best]/60:.2f} min/ckpt", flush=True)

    # Copy the winning probe's output to canonical step_<step>/ so the dispatch
    # phase treats it as cached and doesn't re-evaluate the probe ckpt.
    step, _ = probe_ckpt
    step_str = f"{step:07d}"
    canonical = eval_curve_dir / f"step_{step_str}"
    canonical.mkdir(parents=True, exist_ok=True)
    winner_dir = eval_curve_dir / f"step_{step_str}__probe{best[0]}x{best[1]}"
    winner_metrics = winner_dir / "metrics.json"
    if winner_metrics.exists() and not (canonical / "metrics.json").exists():
        import shutil as _sh
        _sh.copy2(winner_metrics, canonical / "metrics.json")
        print(f"[eval-curve][probe] copied winner metrics → {canonical}/metrics.json",
              flush=True)

    return best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--task-group", default=None, help="override cfg.eval.task_group")
    p.add_argument("--every", type=int, default=None,
                   help="only eval ckpts whose step is a multiple of this (default: all valid)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="lighteval max_samples cap for fast smoke; default = full")
    p.add_argument("--parallel-jobs", type=int, default=None,
                   help="number of concurrent eval workers (default = total_gpus, "
                        "ie one ckpt per GPU; pass 1 for fully sequential 8-GPU DP)")
    p.add_argument("--gpus-per-job", type=int, default=None,
                   help="GPUs per worker; auto = total_gpus/parallel_jobs (default 1)")
    p.add_argument("--probe-strategies", type=str, default=None,
                   help="comma list of NxK strategies to benchmark on FIRST ckpt before "
                        "committing, e.g. '1x8,2x4,4x2,8x1'. Picks the fastest.")
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
    total_gpus = _detect_gpu_count()

    ckpts = _list_ckpts(ckpt_dir)
    if not ckpts:
        print(f"[eval-curve] no valid ckpts under {ckpt_dir}", file=sys.stderr)
        return 1
    if args.every:
        ckpts = [(s, p) for (s, p) in ckpts if s % args.every == 0]
    if args.skip_final and ckpts:
        ckpts = ckpts[:-1]

    print(f"[eval-curve] exp={cfg.exp_id}  task_group={task_group}  max_samples={max_samples}",
          flush=True)
    print(f"[eval-curve] {len(ckpts)} ckpts to eval: steps={[s for (s, _) in ckpts]}", flush=True)
    print(f"[eval-curve] available GPUs={total_gpus}", flush=True)
    if args.dry_run:
        return 0

    # ── PROBE PHASE (optional) ───────────────────────────────────────────
    if args.probe_strategies:
        strategies = [_parse_strategy(s) for s in args.probe_strategies.split(",")]
        winner_n, winner_k = _do_probe(
            probe_ckpt=ckpts[0],
            eval_curve_dir=eval_curve_dir,
            task_group=task_group,
            max_samples=max_samples,
            strategies=strategies,
            keep_hf=True,
            total_gpus=total_gpus,
        )
        print(f"\n[eval-curve][probe] WINNER: {winner_n} workers × {winner_k} GPU/worker",
              flush=True)
        args.parallel_jobs = winner_n
        args.gpus_per_job = winner_k

    # ── DISPATCH PHASE ────────────────────────────────────────────────────
    # Auto-defaulting rule (when args not given):
    #   parallel_jobs = total_gpus   (one ckpt per GPU = max parallelism)
    #   gpus_per_job  = 1            (each ckpt single-GPU evals)
    # For a 1.7B model this is ~5x faster than 1×8-GPU DP because per-task
    # lighteval setup overhead dominates compute. Override explicitly if needed.
    n_workers = args.parallel_jobs if args.parallel_jobs is not None else total_gpus
    k_per = args.gpus_per_job if args.gpus_per_job is not None else max(1, total_gpus // n_workers)
    if n_workers * k_per > total_gpus:
        print(f"[eval-curve] ERROR: parallel_jobs({n_workers}) × gpus_per_job({k_per}) "
              f"> total_gpus({total_gpus})", file=sys.stderr)
        return 2
    groups = _gpu_groups(parallel_jobs=n_workers, gpus_per_job=k_per)
    print(f"[eval-curve] dispatch: {n_workers} workers × {k_per} GPU = {n_workers*k_per}/{total_gpus} GPUs",
          flush=True)

    # Build job list, skipping cached ckpts. Convert nanotron→HF in parallel
    # (4 workers, mostly disk I/O — cuts 21 ckpts × ~1.5 min = 30 min serial
    # down to ~8 min wall).
    jobs: list[EvalJob] = []
    cached_rows: list[dict] = []
    pending_convert: list[tuple[int, Path, Path, Path]] = []
    for step, nano_ckpt in ckpts:
        step_str = f"{step:07d}"
        out_dir = eval_curve_dir / f"step_{step_str}"
        if (cached := _read_metrics(out_dir)) is not None:
            print(f"[eval-curve] step {step}: cached ({len(cached)} scores, "
                  f"macro={_curve_macro(cached):.4f})", flush=True)
            cached_rows.append({"step": step, "macro": _curve_macro(cached), **cached})
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        hf_dst = out_dir / "hf_ckpt"
        pending_convert.append((step, nano_ckpt, hf_dst, out_dir))

    if pending_convert:
        import concurrent.futures as cf
        print(f"[eval-curve] converting {len(pending_convert)} nanotron ckpts → HF "
              f"(4-way parallel)", flush=True)
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {
                ex.submit(_convert_to_hf, nano, hf): (step, nano, hf, out)
                for (step, nano, hf, out) in pending_convert
            }
            for fut in cf.as_completed(futs):
                step, nano, hf, out = futs[fut]
                rc = fut.result()
                if rc != 0:
                    print(f"[eval-curve] HF convert FAILED for step {step}, rc={rc}",
                          file=sys.stderr)
                    continue
                jobs.append(EvalJob(
                    step=step, nanotron_ckpt=nano, hf_dst=hf, out_dir=out,
                    task_group=task_group, max_samples=max_samples,
                    keep_hf=args.keep_hf,
                ))
        print(f"[eval-curve] convert done in {(time.time()-t0)/60:.1f} min "
              f"({len(jobs)} ready, {len(pending_convert)-len(jobs)} failed)",
              flush=True)
        # Sort jobs by step ascending to keep log output coherent.
        jobs.sort(key=lambda j: j.step)

    if not jobs:
        print(f"[eval-curve] all {len(ckpts)} ckpts already evaluated", flush=True)
    else:
        t0 = time.time()
        run_results = _dispatch_pool(jobs, groups)
        total_min = (time.time() - t0) / 60
        print(f"\n[eval-curve] dispatch done in {total_min:.1f} min "
              f"(wall) for {len(jobs)} ckpts × {n_workers}-way parallel × {k_per}-GPU",
              flush=True)
        # Optionally clean up HF dirs
        if not args.keep_hf:
            for j in jobs:
                shutil.rmtree(j.hf_dst, ignore_errors=True)

    # ── COLLECT CURVE ────────────────────────────────────────────────────
    curve_rows: list[dict] = list(cached_rows)
    for step, _ in ckpts:
        step_str = f"{step:07d}"
        out_dir = eval_curve_dir / f"step_{step_str}"
        scores = _read_metrics(out_dir)
        if scores is None:
            continue
        if any(r["step"] == step for r in curve_rows):
            continue
        curve_rows.append({"step": step, "macro": _curve_macro(scores), **scores})

    curve_path = eval_curve_dir / "curve.json"
    curve_path.write_text(json.dumps({
        "exp_id": cfg.exp_id,
        "task_group": task_group,
        "global_batch_tokens": cfg.training.global_batch_tokens,
        "rows": sorted(curve_rows, key=lambda r: r["step"]),
    }, indent=2))
    print(f"[eval-curve] DONE — wrote {len(curve_rows)} rows to {curve_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
