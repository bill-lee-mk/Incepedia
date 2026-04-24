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


# ── Qwen2-style 1.7B for FinePhrase replication (Protocol C) ──────────────
#
# This matches the FinePhrase paper's architecture EXACTLY:
#   "We train a 1.7B parameter language model using a Qwen2-style architecture
#    with 28 layers, hidden dimension of 2048, 16 attention heads with 8 KV
#    heads (GQA), intermediate size 6144. Llama 3.2 tokenizer (vocab 128,256)."
#
# Key differences from Qwen3-1.7B (Protocol B, our working architecture):
#   - num_hidden_layers: 28 (vs. 24)              ← deeper
#   - intermediate_size: 6144 (vs. 8192)          ← narrower MLP
#   - rope_theta: 10_000 (vs. 1e6)                ← Qwen2 / Llama2 default
#   - attention_bias: False (vs. True)            ← no QKV bias
#   - max_position_embeddings: 4096 (vs. 2048)    ← FinePhrase trains seq=4096
#   - tokenizer: Llama-3.2 (vs. Qwen 151k)        ← see launcher VOCAB_DEFAULTS
#
# Approximate parameter count: ~1.67B
# Used by: experiments/exp_finephrase_repro_protC_seed42 (and successors).
# See ADR 0007 for the protocol-as-experimental-variable rationale.
QWEN2STYLE_17B_FINEPHRASE_SPEC: dict[str, Any] = {
    "is_qwen2_config": True,

    "hidden_size": 2048,
    "intermediate_size": 6144,         # FinePhrase paper §Appendix
    "num_hidden_layers": 28,           # FinePhrase paper §Appendix
    "num_attention_heads": 16,
    "num_key_value_heads": 8,          # GQA 2:1
    "max_position_embeddings": 4096,   # FinePhrase trains seq_len=4096

    # Qwen2 / Llama2-style RoPE + no bias
    "rope_theta": 10_000.0,
    "attention_bias": False,

    "initializer_range": 0.02,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,

    # Llama-3.2 tokenizer special tokens (vs. Qwen3's bos=1/eos=2)
    "bos_token_id": 128_000,           # Llama-3.2 <|begin_of_text|>
    "eos_token_id": 128_001,           # Llama-3.2 <|end_of_text|>
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


def qwen2style_finephrase_model_config(vocab_size: int) -> dict[str, Any]:
    """Produce the complete `model.model_config` dict for FinePhrase Protocol C."""
    cfg = dict(QWEN2STYLE_17B_FINEPHRASE_SPEC)
    cfg["vocab_size"] = vocab_size
    return cfg


__all__ = [
    "QWEN3_17B_SPEC",
    "QWEN2STYLE_17B_FINEPHRASE_SPEC",
    "qwen3_model_config",
    "qwen2style_finephrase_model_config",
]
