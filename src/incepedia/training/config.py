"""Pydantic schemas for experiment configs.

An experiment is fully defined by a single `experiments/<exp_id>/config.yaml`
that matches `ExperimentConfig` below. The schema enforces:
    - required fields (track, dataset id, model arch, token budget)
    - track-specific validity (Track 1 standalone vs Track 2 cooldown-fork)
    - consistency with ADR 0004 / 0005

Used by:
    - scripts/run_experiment.py (main orchestrator)
    - scripts/index_experiment.py (registry)
    - src/incepedia/training/{backbone,standalone,cooldown_fork}.py
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Track(int, Enum):
    STANDALONE = 1  # Track 1: pure dataset, full from-scratch training
    SEASONING = 2   # Track 2: cooldown-fork from shared backbone


class Stage(str, Enum):
    P1 = "P1"  # reproduce baselines
    P2 = "P2"  # PoC Incepedia v0.1
    P3 = "P3"  # iterate + scale


class DatasetSpec(BaseModel):
    """Which dataset (tokenized) the training reads."""

    id: str                          # e.g. "cosmopedia_v2_reference"
    path: str                        # path under data/datasets/
    tokens: int                      # approximate token count
    notes: str | None = None


class ModelSpec(BaseModel):
    """Architecture spec. See ADR 0007 for the multi-protocol decision.

    - `llama2-1.82B`: Protocol A (reference anchor), aligns with
      Cosmopedia/SmolLM/FineWeb published numbers.
    - `qwen3-1.7B`: Protocol B (working architecture), modern + GQA + Qwen RoPE.
    - `qwen2style-1.7B-finephrase`: Protocol C (FinePhrase replication),
      28-layer Qwen2-style + Llama-3.2 tokenizer to enable direct curve
      comparison with the FinePhrase paper's Figure 1 (see ADR 0007).

    Tokenizer is per-protocol now (was originally shared, but Protocol C
    needs Llama-3.2 to match FinePhrase exactly).
    """

    arch: Literal[
        "llama2-1.82B",
        "qwen3-1.7B",
        "qwen2style-1.7B-finephrase",
    ] = "llama2-1.82B"
    seq_len: int = 2048
    tokenizer: str = "mistralai/Mistral-7B-v0.1"
    vocab_size: int = 32000
    init_from: str | None = None     # path to backbone ckpt (Track 2 only)


class ScheduleSpec(BaseModel):
    """Trapezoidal LR schedule with optional cooldown."""

    scheduler: Literal["trapezoidal", "cosine"] = "trapezoidal"
    lr_max: float = 3.0e-4
    lr_min: float = 0.0
    warmup_tokens: int = 500_000_000   # ~500M tokens warmup
    stable_tokens: int                 # stable phase token count
    cooldown_tokens: int = 0           # 0 for standalone, ~6B for cooldown-fork
    cooldown_from_fraction: float | None = None  # alt: fraction of total


class TrainingSpec(BaseModel):
    train_tokens: int                  # total tokens to see (warmup + stable + cooldown)
    global_batch_tokens: int = 1_310_720   # 1.28M tokens/step ≈ 640 samples × 2048
    micro_batch_size: int = 4
    seed: int = 42
    mixed_precision: Literal["bf16", "fp16"] = "bf16"
    gradient_clip: float = 1.0
    weight_decay: float = 0.1
    schedule: ScheduleSpec


class EvalSpec(BaseModel):
    task_group: Literal[
        "cosmopedia-full",
        "early-signal",
        "math",
        "csr-only",
        "mmlu-only",
    ] = "cosmopedia-full"
    max_samples: int | None = 1000
    eval_every_tokens: int = 2_000_000_000  # eval every 2B tokens during training
    final_eval: bool = True


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration.

    Example (Track 1, standalone reproduction):
        exp_id: exp_ref_cosmopedia_v2_seed42
        stage: P1
        track: 1
        dataset:
          id: cosmopedia_v2_reference
          path: data/datasets/cosmopedia_v2_reference
          tokens: 28_000_000_000
        model:
          arch: llama2-1.82B
        training:
          train_tokens: 30_000_000_000
          seed: 42
          schedule:
            scheduler: trapezoidal
            stable_tokens: 24_000_000_000
            cooldown_tokens: 5_500_000_000
        eval:
          task_group: early-signal
    """

    exp_id: str                        # unique, matches dir name
    stage: Stage
    track: Track
    branch_from: str | None = None     # parent exp_id:ckpt_step for forks
    dataset: DatasetSpec
    model: ModelSpec
    training: TrainingSpec
    eval: EvalSpec = Field(default_factory=EvalSpec)
    notes: str = ""

    @model_validator(mode="after")
    def _validate_track_vs_fields(self) -> "ExperimentConfig":
        if self.track == Track.SEASONING:
            if not self.branch_from and not self.model.init_from:
                raise ValueError(
                    "Track 2 (seasoning) requires either `branch_from` or `model.init_from` "
                    "pointing to a shared backbone checkpoint."
                )
            if self.training.schedule.cooldown_tokens <= 0:
                raise ValueError("Track 2 requires `schedule.cooldown_tokens > 0`.")
        if self.track == Track.STANDALONE:
            if self.branch_from is not None:
                raise ValueError(
                    "Track 1 (standalone) must NOT have `branch_from`; it trains from scratch."
                )
        return self

    @property
    def exp_dir(self) -> Path:
        from incepedia.config import EXPERIMENTS_DIR
        return EXPERIMENTS_DIR / self.exp_id


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiments/<exp>/config.yaml."""
    import yaml
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)
