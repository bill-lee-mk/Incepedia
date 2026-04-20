# incepedia.training

Nanotron training launchers and experiment orchestration.

## Modules

- `config.py` — Pydantic schema for `experiments/*/config.yaml`. Enforces Track 1 vs Track 2 validity.
- `launcher.py` — Renders nanotron YAML from our config, launches via `accelerate launch`. Single function `launch_training(cfg)` handles:
    - **backbone** (shared FineWeb-Edu prefix, no cooldown)
    - **standalone** (Track 1 full run, trapezoidal with cooldown at end)
    - **cooldown-fork** (Track 2 fork from backbone, immediate cooldown)
- `scripts/run_experiment.py` (repo root) — Main orchestrator: config → train → eval → INDEX → NAS sync.

## Llama-1.82B architecture (FineWeb ablation standard)

- 24 layers × 2048 hidden × 16 heads × 8192 MLP
- Full attention (no GQA — matches FineWeb paper's base)
- rope_theta 10000, seq_len 2048
- vocab_size 32000 (Mistral tokenizer)

## Track semantics in config.yaml

```yaml
track: 1                    # Track 1: standalone, from scratch
branch_from: null           # MUST be null for Track 1
model:
  init_from: null           # MUST be null for Track 1
training:
  schedule:
    cooldown_tokens: 5.5e9  # standard trapezoidal cooldown at end

# OR:

track: 2                    # Track 2: cooldown-fork
branch_from: backbone_fineweb_edu:final
model:
  init_from: experiments/backbone_fineweb_edu/ckpt/step_15000   # REQUIRED
training:
  schedule:
    cooldown_tokens: 6e9    # REQUIRED, > 0
    stable_tokens: 0        # cooldown-fork enters cooldown immediately
```

## Why launcher.py lazy-imports nanotron

Eval-only and data-prep paths should work even when `nanotron + flash_attn` aren't
installed. We import nanotron only inside `launch_training()` so `from incepedia.training
import config` is always safe.

## Running a full experiment

```bash
python scripts/run_experiment.py --config experiments/exp_ref_cosmopedia_v2_seed42/config.yaml
```

This will:
1. Sync `config.yaml` to NAS immediately
2. Launch nanotron training (8×H100)
3. Sync checkpoints to NAS
4. Run lighteval early-signal
5. Write `metrics.json`
6. Update `INDEX.parquet`
7. Sync eval outputs to NAS
