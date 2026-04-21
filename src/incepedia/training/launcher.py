"""Nanotron training launcher — Track 1 standalone + Track 2 cooldown-fork + shared backbone.

Single entry point `launch_training(config)` handles all three regimes:
    - backbone       (Track 2's shared prefix; `init_from=None`, `cooldown_tokens=0`)
    - standalone     (Track 1 full run; no backbone, trapezoidal over full train_tokens)
    - cooldown_fork  (Track 2's fork; `init_from=<backbone_ckpt>`, cooldown_tokens>0,
                     the stable phase is effectively 0 here — we enter cooldown
                     immediately on the fork)

Rendering strategy:
    1. Read our `ExperimentConfig` (pydantic)
    2. Generate a nanotron YAML (its native config format)
    3. Invoke `accelerate launch -m nanotron.trainer --config-file <yaml>`
       (programmatic, to avoid nanotron CLI breakages)

Note: this module intentionally does **not** import nanotron at module level.
We want eval/generation paths to work even when nanotron+flash_attn are absent.
nanotron is imported inside `launch_training` lazily.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from incepedia.config import REPO_ROOT
from incepedia.training.config import ExperimentConfig, Track


# ── Architecture specs (see ADR 0007 for dual-protocol decision) ──────

# Protocol A · Llama2-1.82B(reference anchor — aligns with Cosmopedia /
# SmolLM / FineWeb published numbers; full attention, no GQA, RoPE θ=10000).
LLAMA2_182B_SPEC = dict(
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=16,            # full attention (not GQA)
    max_position_embeddings=2048,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    qkv_bias=False,
    hidden_act="silu",
)

# Protocol B · Qwen3-style 1.7B(working architecture — GQA + Qwen RoPE +
# QKV bias; ~1.64B params; uses nanotron's built-in Qwen2Config via the
# `is_qwen2_config: True` flag. See src/incepedia/training/nanotron_qwen.py).
from incepedia.training.nanotron_qwen import QWEN3_17B_SPEC as _QWEN3_SPEC  # noqa: E402

QWEN3_17B_SPEC = dict(_QWEN3_SPEC)  # local copy so launcher is self-contained when read

ARCHITECTURE_SPECS: dict[str, dict] = {
    "llama2-1.82B": LLAMA2_182B_SPEC,
    "qwen3-1.7B": QWEN3_17B_SPEC,
}

# vocab_size defaults per architecture (when config.yaml doesn't set it)
VOCAB_DEFAULTS: dict[str, int] = {
    "llama2-1.82B": 32000,   # Mistral tokenizer
    "qwen3-1.7B": 151936,    # Qwen tokenizer (BBPE, 151k vocab)
}

# back-compat alias for any existing import sites
LLAMA_182B_SPEC = LLAMA2_182B_SPEC


def build_nanotron_yaml(cfg: ExperimentConfig) -> dict:
    """Render our ExperimentConfig into nanotron's expected YAML schema.

    Architecture is dispatched via `cfg.model.arch`:
      - `llama2-1.82B` → stock nanotron Llama config (Protocol A reference)
      - `qwen3-1.7B`   → patched nanotron config with QKV bias + Qwen RoPE
                          (Protocol B working; requires `nanotron_qwen` patch
                          loaded by trainer entry point)
    """
    m = cfg.model
    t = cfg.training
    s = t.schedule

    if m.arch not in ARCHITECTURE_SPECS:
        raise ValueError(
            f"Unknown arch '{m.arch}'. Known: {list(ARCHITECTURE_SPECS)}. "
            f"See ADR 0007 to add new architectures."
        )
    arch_spec = dict(ARCHITECTURE_SPECS[m.arch])
    # Inject vocab_size (from config.yaml, else arch default)
    arch_spec["vocab_size"] = m.vocab_size if m.vocab_size else VOCAB_DEFAULTS[m.arch]

    # Nanotron uses "train_steps" not tokens. Convert:
    steps_train = t.train_tokens // t.global_batch_tokens
    steps_warmup = s.warmup_tokens // t.global_batch_tokens
    steps_stable = s.stable_tokens // t.global_batch_tokens
    steps_cooldown = s.cooldown_tokens // t.global_batch_tokens

    return {
        "general": {
            "project": "incepedia",
            "run": cfg.exp_id,
            "seed": t.seed,
            "step": None,
            "consumed_train_samples": None,
            "ignore_sanity_checks": False,
        },
        "checkpoints": {
            "checkpoint_interval": max(2000, steps_train // 20),  # ~20 checkpoints per run
            "checkpoints_path": str(cfg.exp_dir / "ckpt"),
            "checkpoints_path_is_shared_file_system": False,
            "resume_checkpoint_path": m.init_from,
        },
        "model": {
            "ddp_bucket_cap_mb": 25,
            "dtype": t.mixed_precision,
            "init_method": {"std": arch_spec.get("initializer_range", 0.02)},
            "make_vocab_size_divisible_by": 1,
            # nanotron dispatches between LlamaConfig and Qwen2Config by the
            # presence of `is_qwen2_config: True` inside model_config.
            "model_config": arch_spec,
        },
        "tokenizer": {
            "tokenizer_max_length": m.seq_len,
            "tokenizer_name_or_path": m.tokenizer,
            "tokenizer_revision": None,
        },
        "tokens": {
            "batch_accumulation_per_replica": 1,
            "micro_batch_size": t.micro_batch_size,
            "sequence_length": m.seq_len,
            "train_steps": steps_train,
            "val_check_interval": -1,
        },
        "optimizer": {
            "zero_stage": 1,
            "weight_decay": t.weight_decay,
            "clip_grad": t.gradient_clip,
            "accumulate_grad_in_fp32": True,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1.0e-8,
            "learning_rate_scheduler": {
                "learning_rate": s.lr_max,
                "lr_warmup_steps": steps_warmup,
                "lr_warmup_style": "linear",
                "lr_decay_steps": steps_cooldown if steps_cooldown > 0 else 0,
                "lr_decay_starting_step": steps_warmup + steps_stable,
                "min_decay_lr": s.lr_min,
                "lr_decay_style": "linear" if s.scheduler == "trapezoidal" else "cosine",
            },
        },
        "data": {
            "dataset": {
                "dataset_overwrite_cache": False,
                "dataset_processing_num_proc_per_process": 8,
                "hf_dataset_or_datasets": cfg.dataset.path,
                "hf_dataset_splits": "train",
                "text_column_name": "text",
            },
            "num_loading_workers": 4,
            "seed": t.seed,
        },
        "logging": {
            "iteration_step_info_interval": 10,
            "log_level": "info",
            "log_level_replica": "info",
            # Aim tracker — local SQLite-backed DB, no cloud deps (ADR re: tracking choice)
            # All runs land under REPO_ROOT/aim/ ; cross-run compare via `aim up`.
            "aim": {
                "repo": str(REPO_ROOT / "aim"),
                "experiment": "incepedia",
                "log_interval": 10,
                "run_hash": cfg.exp_id,   # stable hash → same run id across resumes
            },
        },
        "parallelism": {
            "dp": 8,         # 8 H100s data-parallel
            "pp": 1,
            "tp": 1,
            "pp_engine": "1f1b",
            "tp_linear_async_communication": False,
            "tp_mode": "ALL_REDUCE",
        },
    }


def launch_training(cfg: ExperimentConfig, num_processes: int = 8, dry_run: bool = False) -> int:
    """Render + launch nanotron training.

    Returns the subprocess returncode (0 = success).
    """
    # Lazy import — only require nanotron for actual launches, not dry-runs
    if not dry_run:
        try:
            import nanotron  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "nanotron not installed. Run `pip install -e third_party/nanotron` "
                "and ensure flash-attn is available. "
                f"(underlying error: {e})"
            )

    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    (cfg.exp_dir / "ckpt").mkdir(exist_ok=True)
    nt_yaml = build_nanotron_yaml(cfg)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(nt_yaml, f)
        yaml_path = Path(f.name)

    cmd = [
        "accelerate", "launch",
        f"--num_processes={num_processes}",
        "--mixed_precision", cfg.training.mixed_precision,
        "-m", "nanotron.trainer",
        "--config-file", str(yaml_path),
    ]
    print(f"[train] exp={cfg.exp_id}  track={cfg.track}  "
          f"tokens={cfg.training.train_tokens:,}", file=sys.stderr)
    print("[train] command:", " ".join(cmd), file=sys.stderr)
    if dry_run:
        print("[train] dry-run — not executing", file=sys.stderr)
        return 0

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    print(f"[train] exited with code {result.returncode} in {dt/60:.1f} min", file=sys.stderr)
    return result.returncode
