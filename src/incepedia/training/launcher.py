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

import os as _os
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
    # Flag: tells nanotron this is a Llama family model (triggers LlamaModel).
    is_llama_config=True,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=16,            # full attention (not GQA)
    max_position_embeddings=2048,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    attention_bias=False,              # nanotron LlamaConfig uses `attention_bias`
    hidden_act="silu",
)

# Protocol B · Qwen3-style 1.7B(working architecture — GQA + Qwen RoPE +
# QKV bias; ~1.64B params; uses nanotron's built-in Qwen2Config via the
# `is_qwen2_config: True` flag. See src/incepedia/training/nanotron_qwen.py).
from incepedia.training.nanotron_qwen import QWEN3_17B_SPEC as _QWEN3_SPEC  # noqa: E402
from incepedia.training.nanotron_qwen import QWEN2STYLE_17B_FINEPHRASE_SPEC as _QWEN2STYLE_SPEC  # noqa: E402

QWEN3_17B_SPEC = dict(_QWEN3_SPEC)  # local copy so launcher is self-contained when read

# Protocol C · Qwen2-style 1.7B + Llama-3.2 tokenizer (FinePhrase paper, ADR 0007)
# Same family as Protocol B (Qwen2 model class) but different hyperparams &
# tokenizer to enable direct reproduction of FinePhrase Figure 1 curves.
QWEN2STYLE_17B_FINEPHRASE_SPEC = dict(_QWEN2STYLE_SPEC)

ARCHITECTURE_SPECS: dict[str, dict] = {
    "llama2-1.82B": LLAMA2_182B_SPEC,
    "qwen3-1.7B": QWEN3_17B_SPEC,
    "qwen2style-1.7B-finephrase": QWEN2STYLE_17B_FINEPHRASE_SPEC,
}

# vocab_size defaults per architecture (when config.yaml doesn't set it).
# IMPORTANT: must match the *actual* tokenizer's vocab size reported by
# transformers — nanotron asserts model.vocab_size == dataset.vocab_size where
# dataset.vocab_size comes from the tokenizer's `get_vocab()` size recorded in
# the datatrove metadata.
VOCAB_DEFAULTS: dict[str, int] = {
    "llama2-1.82B": 32000,   # Mistral tokenizer
    "qwen3-1.7B": 151665,    # Qwen/Qwen2.5-1.5B tokenizer: len(get_vocab()) = 151665
    "qwen2style-1.7B-finephrase": 128256,  # Llama-3.2 tokenizer (FinePhrase paper)
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

    # Compute gradient accumulation needed to hit the configured global batch.
    # CRITICAL: nanotron's effective tokens-per-step is
    #     micro_batch_size * batch_accumulation_per_replica * seq_len * dp
    # We MUST set `batch_accumulation_per_replica` to honour
    # `t.global_batch_tokens`, otherwise we accidentally train on
    # `(steps_train * mbs * seq * dp)` tokens — a 20× under-shoot was
    # observed on 2026-04-22 with the previous hard-coded value of 1 (seed42
    # nominally 30B trained 1.5B).  Dp is hard-coded to 8 below; if you change
    # parallelism, also revisit this calculation.
    dp = 8
    tokens_per_microstep = t.micro_batch_size * m.seq_len * dp
    if t.global_batch_tokens % tokens_per_microstep != 0:
        raise ValueError(
            f"global_batch_tokens={t.global_batch_tokens} is not divisible by "
            f"micro_batch_size×seq_len×dp = {t.micro_batch_size}×{m.seq_len}×{dp} "
            f"= {tokens_per_microstep}. Adjust micro_batch_size or global_batch_tokens."
        )
    grad_accum = t.global_batch_tokens // tokens_per_microstep

    # Resolve the tokenized dataset folder.  Config holds a relative path;
    # nanotron needs an absolute path to the directory that contains the
    # datatrove `.ds`/`.ds.index`/`.ds.metadata` shards.
    dataset_folder = Path(cfg.dataset.path)
    if not dataset_folder.is_absolute():
        dataset_folder = REPO_ROOT / dataset_folder
    if (dataset_folder / "tokenized").is_dir():
        dataset_folder = dataset_folder / "tokenized"

    # ── Auto-resume checkpoint resolution ──
    # Priority:
    #   1) explicit:  cfg.model.init_from is set in YAML → obey it
    #   2) auto:      walk `<exp_dir>/ckpt/<step>/` for the latest VALID ckpt
    #                 (has config.yaml + model/ + checkpoint_metadata.json)
    #   3) fresh:     no valid ckpt found, or INCEPEDIA_FRESH_START=1
    # This makes long runs survive SSH drops / crashes: just re-launch, we
    # resume from the last save (interval ≈ every 2000 steps, see below).
    resume_path: str | None = None
    if m.init_from:
        resume_path = str(m.init_from)
        print(f"[launcher] RESUME: explicit init_from={resume_path}", file=sys.stderr)
    elif _os.environ.get("INCEPEDIA_FRESH_START") != "1":
        ckpt_dir = cfg.exp_dir / "ckpt"
        valid = []
        if ckpt_dir.is_dir():
            for p in ckpt_dir.iterdir():
                if not (p.is_dir() and p.name.isdigit()):
                    continue
                if ((p / "config.yaml").is_file()
                        and (p / "model").is_dir()
                        and (p / "checkpoint_metadata.json").is_file()):
                    valid.append((int(p.name), p))
        if valid:
            valid.sort()
            step_num, path = valid[-1]
            resume_path = str(path)
            print(f"[launcher] AUTO-RESUME from step {step_num} "
                  f"({path})", file=sys.stderr)
            print(f"[launcher]   to override: `INCEPEDIA_FRESH_START=1` "
                  f"or delete {ckpt_dir}", file=sys.stderr)
    if resume_path is None:
        print(f"[launcher] FRESH start (no ckpt found / override requested)",
              file=sys.stderr)

    # Map our short precision code to the names nanotron accepts via its
    # str→torch.dtype hook.
    dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    model_dtype = dtype_map.get(t.mixed_precision, t.mixed_precision)

    return {
        "general": {
            "project": "incepedia",
            "run": cfg.exp_id,
            "seed": t.seed,
            "step": None,
            "consumed_train_samples": None,
            # Nanotron's per-step cross-rank `assert_tensor_synced_across_pg`
            # runs `torch.testing.assert_close` on full tensors — measured via
            # py-spy to consume ~1s/step on this 8×H100 setup (≈2× slowdown).
            # Upstream default is True, and nanotron prints a warning if False.
            # Keep it on the default; re-enable by setting this to False only
            # when debugging divergence / NaN / rank desync.
            "ignore_sanity_checks": True,
        },
        "checkpoints": {
            "checkpoint_interval": max(2000, steps_train // 20),  # ~20 checkpoints per run
            "checkpoints_path": str(cfg.exp_dir / "ckpt"),
            "checkpoints_path_is_shared_file_system": False,
            # See auto-resume block above for how this value is populated.
            "resume_checkpoint_path": resume_path,
        },
        "model": {
            "ddp_bucket_cap_mb": 25,
            "dtype": model_dtype,
            # RandomInit(std) — dacite will build the RandomInit dataclass.
            "init_method": {"std": arch_spec.get("initializer_range", 0.02)},
            "make_vocab_size_divisible_by": 1,
            # nanotron dispatches between LlamaConfig and Qwen2Config by the
            # presence of `is_llama_config` / `is_qwen2_config` inside
            # model_config.  The spec dict we store already sets the right flag.
            "model_config": arch_spec,
        },
        "tokenizer": {
            "tokenizer_max_length": m.seq_len,
            "tokenizer_name_or_path": m.tokenizer,
            "tokenizer_revision": None,
        },
        "tokens": {
            "batch_accumulation_per_replica": grad_accum,
            "micro_batch_size": t.micro_batch_size,
            "sequence_length": m.seq_len,
            "train_steps": steps_train,
            "val_check_interval": -1,
            "limit_val_batches": 0,
            "limit_test_batches": 0,
        },
        "optimizer": {
            # zero_stage=0 (pure DP). 1.7B fits easily per H100 in bf16.
            # Nanotron's ZeRO-1 + fp32-grad-accum path hits an unfinished
            # `reduce_scatter` branch in `gradient_accumulator.get_fp32_accum_hook`
            # (raises NotImplementedError).  Staying at stage 0 sidesteps it
            # and matches Cosmo-1B's published setup.
            "zero_stage": 0,
            "weight_decay": t.weight_decay,
            "clip_grad": t.gradient_clip,
            "accumulate_grad_in_fp32": True,
            "optimizer_factory": {
                "name": "adamW",
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_eps": 1.0e-8,
                "torch_adam_is_fused": True,
            },
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
        # Nanotron expects `data_stages: List[DatasetStageArgs]`.  We create a
        # single stable stage starting at step 1 pointing to the NanosetDatasetsArgs
        # (datatrove `.ds` shard folder).  Aim/TensorBoard logging is handled
        # separately (post-processed from the checkpoint logs); the top-level
        # `logging` key in nanotron only controls verbosity.
        "data_stages": [
            {
                "name": "stable",
                "start_training_step": 1,
                "data": {
                    "dataset": {
                        "dataset_folder": str(dataset_folder),
                    },
                    "num_loading_workers": 4,
                    "seed": t.seed,
                },
            }
        ],
        "logging": {
            "iteration_step_info_interval": 10,
            "log_level": "info",
            "log_level_replica": "info",
        },
        "parallelism": {
            "dp": 8,         # 8 H100s data-parallel
            "pp": 1,
            "tp": 1,
            "pp_engine": "1f1b",
            "tp_linear_async_communication": False,
            "tp_mode": "ALL_REDUCE",
            "expert_parallel_size": 1,
            "context_parallel_size": 1,
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

    # Persist the rendered YAML alongside the experiment so crashes can be
    # reproduced from disk (and not from `/tmp/` which may be cleared).
    yaml_path = cfg.exp_dir / "nanotron.yaml"
    yaml_path.write_text(yaml.safe_dump(nt_yaml))

    # Nanotron's CLI entry is `run_train.py` at the repo root, NOT the
    # `nanotron.trainer` module (that only defines classes — invoking it with
    # `-m` loads and exits cleanly with no training).  We vendor nanotron under
    # `third_party/nanotron/`, so we call that file directly.
    run_train_py = REPO_ROOT / "third_party" / "nanotron" / "run_train.py"
    if not run_train_py.exists():
        raise FileNotFoundError(
            f"Cannot find nanotron entry script at {run_train_py}. "
            "Verify third_party/nanotron is present (git submodule / clone)."
        )
    cmd = [
        "accelerate", "launch",
        f"--num_processes={num_processes}",
        "--mixed_precision", cfg.training.mixed_precision,
        str(run_train_py),
        "--config-file", str(yaml_path),
    ]
    train_log = cfg.exp_dir / "train.log"
    print(f"[train] exp={cfg.exp_id}  track={cfg.track}  "
          f"tokens={cfg.training.train_tokens:,}", file=sys.stderr)
    print("[train] command:", " ".join(cmd), file=sys.stderr)
    print(f"[train] log    : {train_log} (tee; also echoed to stderr)", file=sys.stderr)
    if dry_run:
        print("[train] dry-run — not executing", file=sys.stderr)
        return 0

    t0 = time.time()
    # Tee child stdout/stderr to both console and `experiments/<exp_id>/train.log`
    # so a multi-rank silent crash still leaves on-disk evidence.
    with train_log.open("ab", buffering=0) as log_fh:
        header = (
            f"# ==== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"launch {cfg.exp_id} ====\n"
            f"# cmd: {' '.join(cmd)}\n"
        ).encode()
        log_fh.write(header)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.readline(), b""):
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()
            log_fh.write(chunk)
        returncode = proc.wait()
    dt = time.time() - t0
    print(f"[train] exited with code {returncode} in {dt/60:.1f} min "
          f"(log: {train_log})", file=sys.stderr)
    return returncode
