# 0009 · Always materialise an HF-format ckpt before evaluation / publishing

- **Status**: accepted (implementation landed 2026-04-22)
- **Date**: 2026-04-22
- **Deciders**: bill-lee-mk

## Context

Nanotron writes its checkpoints in a custom sharded layout:

```
experiments/<exp_id>/ckpt/<step>/
├── config.yaml                    # nanotron training config snapshot
├── checkpoint_metadata.json       # step counter, consumed samples, etc.
├── model_config.json              # arch dict (Qwen2 / Llama config)
├── model/<submodule>/...
│   └── pp_block/.../model_weight_pp-rank-A-of-B_tp-rank-X-of-Y.safetensors
├── optimizer/...
├── lr_scheduler
└── random
```

This format is great for distributed-training resume but **incompatible with**:

* `lighteval` `accelerate` / `transformers` backend — wants `Qwen2ForCausalLM`-style
  `config.json` + single-file `model.safetensors`.
* `lm-evaluation-harness` — same.
* `vllm`, `sglang`, `text-generation-inference` — same.
* `huggingface_hub.upload_folder` for public release — same.

Lighteval 0.13 *does* ship a `main_nanotron` path, but on 2026-04-22 it has
**at least 6 cascading bugs** (`KeyError TaskID`, `Doc.context` missing,
`SampleCache` writing parquet inside the model dir, `model_dump()` called on a
plain `@dataclass`, `Doc.stop_sequence` vs `stop_sequences`, etc.).  Every fix
exposes another upstream issue.  The lighteval-nanotron integration drifted
during the lighteval 0.13 rewrite and is currently unmaintained.

## Decision

For every nanotron checkpoint we want to evaluate or publish, we **first
convert it to HuggingFace `transformers` format** using a small in-tree
converter, then run all downstream tools (eval, inference, publishing) on the
HF copy.  Nanotron is purely a training runtime; it never participates in
evaluation.

```
nanotron training        scripts/convert_nanotron_qwen2_to_hf.py
   (3D parallel)              ↓ one-shot, ~30s on CPU
       │
       ▼                experiments/<exp>/hf_ckpt/
experiments/<exp>/         ├── config.json    (Qwen2ForCausalLM)
   ckpt/<step>/            ├── model.safetensors  (3.5 GB single file)
                           ├── tokenizer.json
                           └── generation_config.json
                                  │
                ┌─────────────────┼──────────────────┬─────────────────┐
                ▼                 ▼                  ▼                 ▼
          lighteval         lm-evaluation-       vllm /           HF Hub upload
        (accelerate)         harness            sglang             (P3 release)
```

## Implementation

* `scripts/convert_nanotron_qwen2_to_hf.py` (~150 LOC)
  * Reads `<ckpt>/model_config.json` for arch dimensions
  * Walks the nanotron `model/...` subtree and reassembles tensors:
    * QKV: split fused `[Q;K;V]` weight + bias (rows partitioned by Q/KV head dims)
    * MLP: split fused `[gate; up]` weight (rows partitioned by `intermediate_size`)
    * Layer norms / o_proj / down_proj / token_embedding / lm_head: copy as-is
  * Writes a HF Qwen2 `config.json` (preserves `attention_bias=True`,
    `rope_theta=1e6` etc. from training spec)
  * Saves single-file `model.safetensors` and downloads the matching tokenizer
    via `transformers.AutoTokenizer.save_pretrained`
* Orchestrator (`scripts/run_experiment.py`) auto-runs the converter immediately
  after training succeeds, **before** the eval phase.  In `--eval-only` mode,
  it converts on demand if `hf_ckpt/` is missing.
* `scripts/sync_to_nas.sh hf_ckpt <exp_id>` syncs the converted ckpt to NAS
  (~3.5 GB vs ~28 GB raw nanotron — a 8x reduction).

## Why this is the industry-standard approach

* **HF SmolLM team**: trains in nanotron → converts to HF → evaluates with
  lighteval-accelerate / lm-eval-harness → publishes HF format.  They do
  *not* use lighteval-nanotron path.
* **AllenAI OLMo**: trains in OLMo-core → converts to HF → publishes HF.
* **Microsoft Phi**: trains internally → publishes HF format.
* The pattern is **train in your fastest framework, then convert to the
  community standard for all downstream consumption**.

## Alternatives considered

1. **Patch lighteval-nanotron path until it works**.  Tried 6+ monkey patches,
   each one exposed the next.  High maintenance, fragile, single-tool lock-in.
   **Rejected**.
2. **Add Qwen2 support to nanotron's own `convert_nanotron_to_hf.py`** (only
   has Llama + Mamba upstream).  Equivalent to what we did, but as an upstream
   PR.  Worth doing eventually for community contribution; for now the
   in-tree script is faster to iterate on.
3. **Skip conversion, write our own bare-bones evaluator** that loads nanotron
   weights directly.  Reinvents lighteval task definitions / metrics; gives up
   the 129-task Cosmopedia parity.  **Rejected**.

## Consequences

### Positive

* Eval pipeline is the **same code path Cosmopedia / SmolLM use** → scores
  are directly comparable to their public numbers (ADR 0004 P1 verification
  criterion ±0.5pp).
* Any future evaluator (vllm, sglang, lm-eval-harness, custom downstream
  tasks) plugs in for free.
* Per-experiment HF ckpt (~3.5 GB) is small enough to push to HF Hub in P3.
* Decouples training framework choice from evaluation framework choice; we
  can switch from nanotron to TorchTitan later without touching the eval stack.

### Negative / trade-offs

* Extra ~30 s per ckpt for conversion (negligible vs ~55 h training).
* Extra ~3.5 GB disk per ckpt (negligible vs ~28 GB nanotron ckpt).
* Two storage formats to keep in sync — orchestrator handles this.
* Conversion code is Qwen2-specific; for Llama2-1.82B (Protocol A) we would
  use nanotron's upstream `convert_nanotron_to_hf.py` (already supports Llama).

### Follow-ups

* Add Llama-family support to our converter when Protocol A actually trains
  (or use upstream `convert_nanotron_to_hf.py` from `examples/llama/`).
* Once nanotron supports tp/pp > 1 for our experiments, the converter must
  gather TP shards before splitting QKV; today we hard-assume `tp=1, pp=1`.
* Consider upstreaming Qwen2 support to nanotron after stable.

## Related

* `scripts/convert_nanotron_qwen2_to_hf.py`
* `scripts/run_experiment.py` (orchestrator wiring at the train→eval boundary)
* `scripts/sync_to_nas.sh hf_ckpt`
* ADR 0004 (evaluation protocol — defines "publication-grade" claim)
* ADR 0006 (lighteval pinning policy)
* ADR 0007 (dual-protocol architecture — defines what we evaluate)
* `docs/multi-machine-eval-setup.md` (how 164 A100 reads HF ckpts from NAS)
