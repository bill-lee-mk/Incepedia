#!/usr/bin/env python
"""Lint nanotron training YAML: list all non-default values we are setting.

Why
---
Nanotron has a number of *performance-critical* config fields whose defaults
are sensible but whose opposite value silently enables a slow / debug code
path.  The canonical example we hit on 2026-04-21 was
`general.ignore_sanity_checks=False`: upstream default is `True`, but our
launcher set it to `False`, which enabled a per-step `torch.testing.assert_close`
across all ranks — costing ~50 % throughput.  There was no crash, no warning
loud enough to notice in logs, and `nvidia-smi` looked fine.

What this script does
---------------------
1. Run the launcher in `dry_run` mode (or read a pre-rendered YAML) to obtain
   the full nanotron config dict our code emits.
2. Load each nanotron dataclass's declared defaults from the installed
   `nanotron.config` module.
3. Diff our emitted dict against defaults recursively and print:
     * ❌ fields we set to a value DIFFERENT from the nanotron default
     * ⚠  fields we set that do NOT exist in nanotron's current schema
          (= we or nanotron drifted)
     * ✓  fields we set to the same value as the default (silently allowed)
4. Exit code 0 if the "non-default" report matches an allow-list
   (`scripts/nanotron_yaml_allowlist.yaml`), or prints the diff for human review.

Usage
-----
    # audit a rendered yaml (e.g. the one the launcher just wrote):
    python scripts/lint_nanotron_yaml.py --yaml experiments/<exp>/nanotron.yaml

    # or run the launcher dry-run ourselves on an experiment config:
    python scripts/lint_nanotron_yaml.py --config experiments/<exp>/config.yaml

    # print EVERY difference, not just the ones not on the allow-list:
    python scripts/lint_nanotron_yaml.py --config <path> --verbose
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

from incepedia.config import REPO_ROOT


# ──────────────────────────────────────────────────────────────────────────
# Nanotron schema reflection: pull the declared default for every field of
# every nested dataclass in nanotron.config.

def _iter_fields(cls) -> Iterable[tuple[str, Any, Any]]:
    """Yield (name, default, field_type) for the dataclass's immediate fields."""
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                default = f.default_factory()  # type: ignore[misc]
            except Exception:
                default = "<factory>"
        else:
            default = "<required>"
        yield f.name, default, f.type


def _as_dict(x: Any) -> Any:
    """Normalise dataclass defaults so they can be compared to YAML dicts."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _as_dict(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, list):
        return [_as_dict(v) for v in x]
    if isinstance(x, dict):
        return {k: _as_dict(v) for k, v in x.items()}
    return x


def gather_defaults() -> dict[str, dict[str, Any]]:
    """Return {top_key: {field: default, ...}} for the nanotron Config."""
    from nanotron.config.config import (
        CheckpointsArgs,
        Config,
        DataArgs,
        DatasetStageArgs,
        GeneralArgs,
        LoggingArgs,
        ModelArgs,
        OptimizerArgs,
        TokenizerArgs,
        TokensArgs,
    )
    from nanotron.config.parallelism_config import ParallelismArgs

    top_classes: dict[str, Any] = {
        "general": GeneralArgs,
        "parallelism": ParallelismArgs,
        "model": ModelArgs,
        "tokenizer": TokenizerArgs,
        "checkpoints": CheckpointsArgs,
        "logging": LoggingArgs,
        "tokens": TokensArgs,
        "optimizer": OptimizerArgs,
        "data_stages": DatasetStageArgs,   # List[DatasetStageArgs]; audit element fields
    }
    defaults: dict[str, dict[str, Any]] = {}
    for top, cls in top_classes.items():
        defaults[top] = {name: _as_dict(val) for name, val, _ in _iter_fields(cls)}
    # DataArgs lives inside DatasetStageArgs.data; pull it for deeper audit.
    defaults["data_stages._data"] = {name: _as_dict(val) for name, val, _ in _iter_fields(DataArgs)}
    return defaults


# ──────────────────────────────────────────────────────────────────────────
# Diff helpers

def diff_section(ours: dict[str, Any], defaults: dict[str, Any], path: str = "") -> list[dict]:
    """Return a list of {path, kind, ours, default} records.

    kind ∈ {"non_default", "unknown_field"}.
    """
    out: list[dict] = []
    if not isinstance(ours, dict):
        return out
    for k, v in ours.items():
        p = f"{path}.{k}" if path else k
        if k not in defaults:
            out.append({"path": p, "kind": "unknown_field", "ours": v, "default": None})
            continue
        d = defaults[k]
        if isinstance(v, dict) and isinstance(d, dict):
            out.extend(diff_section(v, d, p))
        elif v != d:
            out.append({"path": p, "kind": "non_default", "ours": v, "default": d})
    return out


def render_from_experiment_config(config_yaml: Path) -> dict:
    """Invoke our launcher in dry-run mode to produce the nanotron YAML dict."""
    from incepedia.training.config import load_config
    from incepedia.training.launcher import build_nanotron_yaml

    cfg = load_config(config_yaml)
    return build_nanotron_yaml(cfg)


# Known, *intentionally* non-default fields we accept without review.  Keep
# this list short and annotated; everything outside of it must be either a
# launcher-driven user parameter (hidden via `_is_user_input`) or explained.
INTENTIONAL_NON_DEFAULTS: dict[str, str] = {
    # general
    "general.project": "project name (user param)",
    "general.run": "exp id (user param)",
    "general.seed": "user param",
    # parallelism
    "parallelism.dp": "user param (dp replicas)",
    "parallelism.pp": "user param (pp stages)",
    "parallelism.tp": "user param (tp replicas)",
    "parallelism.pp_engine": "explicit 1f1b",
    "parallelism.tp_mode": "explicit ALL_REDUCE for dp-only setup",
    "parallelism.tp_linear_async_communication": "False when tp=1",
    # model
    "model.dtype": "user param (bf16)",
    "model.init_method": "user param (RandomInit std from arch spec)",
    "model.model_config": "user param — arch spec dict",
    # tokenizer
    "tokenizer.tokenizer_max_length": "user param (seq_len)",
    "tokenizer.tokenizer_name_or_path": "user param",
    # checkpoints
    "checkpoints.checkpoint_interval": "derived from train_steps (user param)",
    "checkpoints.checkpoints_path": "user param (exp_dir/ckpt)",
    "checkpoints.resume_checkpoint_path": "user param (init_from)",
    # logging
    "logging.log_level": "info",
    "logging.log_level_replica": "info",
    "logging.iteration_step_info_interval": "10 (human-readable cadence)",
    # tokens
    "tokens.micro_batch_size": "user param",
    "tokens.sequence_length": "user param",
    "tokens.train_steps": "user param",
    "tokens.batch_accumulation_per_replica": "user param (1)",
    # optimizer
    "optimizer.zero_stage": "0 (avoids nanotron ZeRO-1 fp32-accum NotImplementedError, see commit 04c1457)",
    "optimizer.weight_decay": "user param",
    "optimizer.clip_grad": "user param",
    "optimizer.accumulate_grad_in_fp32": "True — standard stable setting",
    "optimizer.optimizer_factory": "adam beta/eps/fused — standard",
    "optimizer.learning_rate_scheduler": "user param (trapezoidal)",
    # data_stages (first stage)
    "data_stages[0].name": "'stable' — single-stage run",
    "data_stages[0].start_training_step": "1",
    "data_stages[0].data": "user param — Nanoset folder + loaders + seed",
    "data_stages[0].data.dataset": "user param — Nanoset folder",
    "data_stages[0].data.num_loading_workers": "4 (was 1 default; tuned for our shard count)",
    "data_stages[0].data.seed": "user param",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--yaml", type=Path, help="path to an already-rendered nanotron yaml")
    src.add_argument("--config", type=Path, help="path to an experiment config.yaml (launcher dry-run)")
    ap.add_argument("--verbose", action="store_true", help="also print fields that match defaults + allow-listed diffs")
    args = ap.parse_args()

    if args.yaml:
        ours = yaml.safe_load(args.yaml.read_text())
    else:
        ours = render_from_experiment_config(args.config)

    defaults = gather_defaults()

    # Top-level diff
    all_records: list[dict] = []
    for top in ours.keys():
        if top == "data_stages":
            # Audit the first stage only (others would follow the same schema).
            stage = ours["data_stages"][0] if ours["data_stages"] else {}
            all_records += diff_section(stage, defaults["data_stages"], "data_stages[0]")
            if "data" in stage:
                all_records += diff_section(stage["data"], defaults["data_stages._data"], "data_stages[0].data")
            continue
        if top not in defaults:
            all_records.append({"path": top, "kind": "unknown_field", "ours": ours[top], "default": None})
            continue
        all_records += diff_section(ours[top], defaults[top], top)

    # Partition: intentional vs needs-review.
    needs_review: list[dict] = []
    intentional: list[dict] = []
    unknown: list[dict] = []
    for r in all_records:
        if r["kind"] == "unknown_field":
            unknown.append(r)
        elif r["path"] in INTENTIONAL_NON_DEFAULTS:
            intentional.append(r)
        else:
            needs_review.append(r)

    print("── nanotron yaml audit ──────────────────────────────────────────")
    print(f"  source : {args.yaml or args.config}")
    print(f"  total non-default fields: {len(all_records)}")
    print(f"    ✓ intentional (allow-listed): {len(intentional)}")
    print(f"    ❌ needs review             : {len(needs_review)}")
    print(f"    ⚠ unknown (schema drift)   : {len(unknown)}")
    print()

    if args.verbose and intentional:
        print("── allow-listed non-defaults (ok) ──")
        for r in intentional:
            why = INTENTIONAL_NON_DEFAULTS.get(r["path"], "")
            print(f"  ✓ {r['path']}  =>  ours={r['ours']!r}  default={r['default']!r}    # {why}")
        print()

    if unknown:
        print("── ⚠ UNKNOWN FIELDS (schema drift — investigate!) ──")
        for r in unknown:
            print(f"  ⚠ {r['path']}  =>  ours={r['ours']!r}")
        print()

    if needs_review:
        print("── ❌ NON-DEFAULT FIELDS NOT ON ALLOW-LIST ──")
        for r in needs_review:
            print(f"  ❌ {r['path']}  =>  ours={r['ours']!r}  default={r['default']!r}")
        print()
        print("Action:")
        print("  • If the override is intentional, add the path to INTENTIONAL_NON_DEFAULTS with a one-liner why.")
        print("  • If it is accidental (like the 2026-04-21 ignore_sanity_checks=False slowdown), fix the launcher.")
        return 1

    print("ok — no unexplained non-default fields found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
