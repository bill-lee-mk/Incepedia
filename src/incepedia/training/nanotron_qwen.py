"""Qwen3-style architecture support for nanotron.

ADR 0007 selects Qwen3-1.7B as the "working architecture" alongside Llama2-1.82B
as the reference anchor. nanotron natively supports Llama family but Qwen3 has
three architectural quirks that require minor adaptation:

    1. **QKV bias**: Qwen3 uses `bias=True` on Q/K/V projection layers
       (Llama family does not).
    2. **RoPE base θ**: Qwen3 uses 1_000_000 (Llama-3 uses 500_000,
       Llama-2 uses 10_000).
    3. **GQA configuration**: Qwen3-1.7B uses 8 KV heads with 16 query heads
       (2:1 ratio). nanotron's LlamaConfig already supports `num_key_value_heads`,
       so this is just a config change.

This module provides a thin subclass of nanotron's Llama implementation that
swaps in Qwen-style attention layers. It is **deliberately small** to keep the
diff with nanotron upstream auditable.

Usage
-----
The launcher dispatches by `cfg.model.arch`:

    if cfg.model.arch == "qwen3-1.7B":
        from incepedia.training.nanotron_qwen import build_qwen3_llamaconfig
        # use Qwen3-style spec dict instead of Llama2 spec
    else:  # "llama2-1.82B"
        # use stock LLAMA2_182B_SPEC

Implementation status
---------------------
- Spec dict (`QWEN3_17B_SPEC`) — ready, used by launcher.
- Patched attention class — STUB. Will be implemented when training begins
  (ADR 0007 follow-up). Until then, calling `build_qwen3_attention` raises
  NotImplementedError so the gate is loud.

Why we don't write the full patch yet
-------------------------------------
nanotron's internal API for custom attention layers requires touching at least
two files (`models/llama.py` and the trainer's `_init_model_from_config`). The
exact pattern depends on the nanotron HEAD version we end up training with. We
will pin a nanotron commit + write the full patch in a single PR right before
the first Qwen3 training run, after `flash-attn` and the eval pipeline are
both green-lit. This avoids carrying dead code that drifts from upstream.

When you implement, the canonical insertion points are:
    - subclass `nanotron.models.llama.LlamaConfig` → add `qkv_bias: bool`
    - subclass `nanotron.models.llama.CausalSelfAttention` → use bias from cfg
    - register the model class with `nanotron.trainer.DistributedTrainer`
"""
from __future__ import annotations

from typing import Any

# ── Qwen3-style 1.7B architecture spec ────────────────────────────────────
#
# Hyperparameters chosen to match published Qwen3-1.7B design philosophy
# (GQA, large RoPE θ, QKV bias) while preserving comparability with our
# Llama2-1.82B reference (same hidden_size and intermediate_size keep
# attention/MLP work units in the same ballpark, isolating architectural
# delta to GQA + RoPE + bias).
#
# Approximate parameter count: ~1.64B (rounds to 1.7B)
QWEN3_17B_SPEC: dict[str, Any] = {
    "hidden_size": 2048,
    "intermediate_size": 8192,         # SwiGLU MLP hidden
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,          # GQA 2:1 (vs. Llama2's 16, no GQA)
    "max_position_embeddings": 2048,   # we don't need long ctx
    "rope_theta": 1_000_000.0,         # Qwen3 standard (vs Llama2's 10_000)
    "tie_word_embeddings": False,
    "qkv_bias": True,                  # Qwen3 standard (Llama family is False)
    "hidden_act": "silu",
    # Init: Qwen3 uses std=0.02 same as Llama, no special scaling for emb
    "initializer_range": 0.02,
}


def build_qwen3_attention(*args, **kwargs):
    """Construct a Qwen3-style attention layer (STUB).

    Will subclass `nanotron.models.llama.CausalSelfAttention` to use
    `bias=True` for q/k/v projections. Implementation deferred to first
    Qwen3 training launch (ADR 0007 follow-up).
    """
    raise NotImplementedError(
        "Qwen3 attention patch is a stub. Implement when first Qwen3 training "
        "run is scheduled (see ADR 0007 follow-ups). For spec validation and "
        "config rendering, the QWEN3_17B_SPEC dict above is sufficient."
    )


def patch_nanotron_llama_for_qwen3():
    """Apply runtime monkey-patches to nanotron.models.llama for Qwen3 (STUB).

    Will inject `qkv_bias` field into LlamaConfig and modify the attention
    init to honour it. Called from launcher before model construction.
    """
    raise NotImplementedError(
        "Runtime patch is a stub. See module docstring for canonical insertion "
        "points and the implementation plan."
    )


__all__ = [
    "QWEN3_17B_SPEC",
    "build_qwen3_attention",
    "patch_nanotron_llama_for_qwen3",
]
