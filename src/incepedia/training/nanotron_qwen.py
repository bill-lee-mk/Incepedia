"""Qwen3-style architecture support for nanotron.

DISCOVERY: nanotron already ships a full `Qwen2Config` / `Qwen2Model`
implementation under `nanotron.models.qwen` + `nanotron.config.models_config`.
It natively supports:

    - `attention_bias: True`  (the QKV bias that distinguishes Qwen from Llama)
    - `rope_theta: 1_000_000`  (Qwen3 default)
    - GQA via `num_key_value_heads`
    - RMS norm, SwiGLU, rotary embeddings (same as Llama)

Qwen3 = Qwen2 architecture + different init + different default hyperparams.
For our from-scratch training, we just instantiate Qwen2Config with Qwen3's
hyperparameters. No custom patch module is needed.

This module exists to:
    1. Declare our canonical Qwen3-1.7B hyperparameters in ONE place.
    2. Produce the exact dict that our launcher injects into nanotron's yaml.
    3. Mark the `is_qwen2_config: True` flag so nanotron instantiates Qwen2Model
       (not LlamaModel) when it parses our yaml.

See:
    - third_party/nanotron/src/nanotron/models/qwen.py
    - third_party/nanotron/src/nanotron/config/models_config.py (Qwen2Config)
"""
from __future__ import annotations

from typing import Any

# ── Qwen3-style 1.7B architecture spec ────────────────────────────────────
#
# Design philosophy:
#   - Match Llama2-1.82B in hidden_size / intermediate_size / num_hidden_layers
#     so compute per token is similar — isolates the architectural delta to
#     {GQA, RoPE θ, QKV bias} (see ADR 0007).
#   - Approximate parameter count: ~1.64B (rounds to "1.7B")
#
# The key-value fields below are passed verbatim to nanotron's yaml
# `model.model_config` section. The `is_qwen2_config: True` flag tells
# nanotron's config loader to build Qwen2Model instead of LlamaModel.
QWEN3_17B_SPEC: dict[str, Any] = {
    # Flag: tells nanotron this is a Qwen2 family model (triggers Qwen2Model)
    "is_qwen2_config": True,

    # Core transformer dimensions
    "hidden_size": 2048,
    "intermediate_size": 8192,         # SwiGLU MLP hidden
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,          # GQA 2:1 (vs. Llama2's no-GQA)
    "max_position_embeddings": 2048,   # we don't need long ctx for 30B-token ablation

    # Qwen3-specific: larger RoPE base + QKV bias
    "rope_theta": 1_000_000.0,         # Qwen3 standard (Llama2 uses 10_000)
    "attention_bias": True,            # Qwen family uses bias on Q/K/V projections

    # Init + activation
    "initializer_range": 0.02,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,

    # Token / vocab (actual vocab_size filled in by launcher from ExperimentConfig.model.vocab_size)
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
}


def qwen3_model_config(vocab_size: int) -> dict[str, Any]:
    """Produce the complete `model.model_config` dict for nanotron yaml.

    Usage in launcher:
        model_config = qwen3_model_config(vocab_size=151936)  # Qwen vocab
    """
    cfg = dict(QWEN3_17B_SPEC)
    cfg["vocab_size"] = vocab_size
    return cfg


__all__ = ["QWEN3_17B_SPEC", "qwen3_model_config"]
