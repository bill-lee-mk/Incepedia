# 0008 · FlashAttention-3 optional backend — behind `NANOTRON_USE_FA3=1`

- **Status**: accepted (implementation landed; default OFF)
- **Date**: 2026-04-21
- **Deciders**: bill-lee-mk

## Context

- Upstream `nanotron 0.4` is hard-wired to FA2:
  `from flash_attn.flash_attn_interface import flash_attn_func` (`nn/attention.py:40`).
- Our environment also has `flash-attn-3 3.0.0` installed (Hopper-optimized,
  ~10–20 % faster step time vs FA2 on H100 bf16).
- FA3 ships in a separate top-level Python package
  (`flash_attn_interface`, **not** `flash_attn.*`) and only exports
  `flash_attn_func`.  It does **not** provide the `layers.rotary`,
  `modules.mha`, `ops.triton.layer_norm`, or `bert_padding` submodules that
  other parts of nanotron depend on.
- FA3's `flash_attn_func` signature **differs** from FA2's:
  - removes `dropout_p` (pretraining uses 0 anyway — fatal only for SFT)
  - adds `q_descale` / `k_descale` / `v_descale`, `softcap`, `pack_gqa`,
    `deterministic`, `sm_margin`
  - keeps `softmax_scale`, `causal`, `window_size`, `return_attn_probs`

## Decision

Add a **feature-flag switch** in `third_party/nanotron/src/nanotron/nn/attention.py`
(tracked as a local patch at `patches/nanotron_fa3_optional_switch.patch`):

- **Default OFF** (`NANOTRON_USE_FA3` unset or `0`):
  - the module-level `flash_attn_func` symbol continues to point at FA2
    (`flash_attn.flash_attn_interface.flash_attn_func`), exactly as upstream.
- **ON** (`NANOTRON_USE_FA3=1`):
  - import `flash_attn_interface.flash_attn_func` (FA3);
  - build a thin adapter `_fa3_adapter(q, k, v, dropout_p=0, softmax_scale,
    causal, window_size, return_attn_probs, ...)` that asserts `dropout_p==0`
    and forwards only the args FA3 supports;
  - rebind the module-level symbol `flash_attn_func` to the adapter, so
    `flash_attention_forward(...)` continues to compile unchanged.

Rotary, layer_norm, varlen, and bert_padding paths **keep using FA2** — FA3
does not ship replacements.

## Why a runtime switch (not a source-level swap)

- Scientific comparability: seed42's baseline ran on FA2 on 2026-04-21;
  flipping to FA3 mid-project would leak a nuisance variable into Cosmopedia
  vs Incepedia deltas.  The switch lets us re-baseline cleanly when we decide
  to adopt FA3.
- Risk containment: if FA3 produces different loss / grad_norm, we can
  instantly fall back to `NANOTRON_USE_FA3=0` without a code change.

## Validation plan (TODO T1 continuation)

Before flipping the default:

1. **Smoke**: `NANOTRON_USE_FA3=1 python scripts/run_experiment.py --config <exp>`
   runs for ≥ 100 steps without error.
2. **Numeric agreement**: fresh FA2 and FA3 runs with `seed=42`, 50 steps,
   same config.  Loss curves agree to **≤1e-3 relative** at step 50.
3. **Throughput**: record `tokens_per_sec_per_gpu` and `model_tflops_per_gpu`
   at steady state (iter 100–200) for both; FA3 target: +10 % or better.
4. **ADR update**: if results pass, flip to `NANOTRON_USE_FA3=1` in
   `scripts/bootstrap_env.sh` env exports (so new shells inherit it) and
   note the switch date in `docs/project-status.md`.

If any step fails, keep FA2 as default and document the failure here.

## Known non-supports

FA3 will **not** be used by:
- `nanotron.nn.rotary.FlashRotaryEmbedding` (still FA2's `layers.rotary`)
- `nanotron.nn.layer_norm` (still FA2's `ops.triton.layer_norm`)
- any `flash_attn_varlen_*` (Qwen2 packed-varlen path in `models/qwen.py`)
- ring_attention / llama3_ring_attention

These are safe because our current experiments (`dp=8`, `tp=1`, `pp=1`, no
sequence packing across docs) take the non-varlen codepath.  A future change
to use varlen packing would need to revisit this decision.

## Related

- `patches/nanotron_fa3_optional_switch.patch`
- `scripts/bootstrap_env.sh` (installs both FA2 and FA3)
- TODO T1 in `docs/project-status.md`
- ADR 0006 (evaluation stack policy)
