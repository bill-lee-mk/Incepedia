#!/usr/bin/env python
"""Internal worker: evaluate ONE HF checkpoint on a fixed GPU subset.

Spawned by `scripts/eval_all_ckpts.py`. The parent process pins this worker
to a subset of GPUs via CUDA_VISIBLE_DEVICES and gives it a unique master_port
so multiple workers don't collide on torchrun's rendezvous port.

This is intentionally a thin wrapper so the parent can supervise lifecycle
(timeout, kill, restart) without having lighteval state leak between ckpts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from incepedia.eval.runner import EvalRunner


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-ckpt", required=True, help="path to HF safetensors ckpt dir")
    p.add_argument("--out-dir", required=True, help="per-step eval output dir")
    p.add_argument("--task-group", required=True)
    p.add_argument("--num-processes", type=int, required=True,
                   help="GPUs to use (must match len(CUDA_VISIBLE_DEVICES))")
    p.add_argument("--master-port", type=int, default=29600)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = EvalRunner(
        model=args.hf_ckpt,
        output_dir=str(out_dir),
        task_group=args.task_group,
        num_processes=args.num_processes,
        main_process_port=args.master_port,
        max_samples=args.max_samples,
    )
    try:
        scores = runner.run()
    except Exception as e:
        print(f"[eval-worker] FATAL: {e!r}", file=sys.stderr)
        return 1

    runner.write_metrics_json(
        out_dir / "metrics.json",
        extra={"gpus_used": args.num_processes, "master_port": args.master_port},
    )
    print(f"[eval-worker] wrote {out_dir/'metrics.json'} ({len(scores)} scores)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
