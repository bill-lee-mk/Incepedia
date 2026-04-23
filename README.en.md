# Incepedia

> [🇨🇳 中文](./README.md) · 🇬🇧 English (current) · [📦 NAS cold-mirror](/lambda/nfs/us-south-2/incepedia_exp_bak/README.en.md)

**Goal**: synthesize a pretraining dataset **Incepedia** that, under the *exact* same ablation protocol used by SmolLM/Cosmopedia, beats Cosmopedia v1/v2 across multiple benchmarks and approaches SmolLM2.

---

## Current state (snapshot 2026-04-23)

🟢 **P1 · Reproduce** — train→convert→eval pipeline is end-to-end live and validated; seed42 30B run is in flight.

| Module | Status |
|--------|--------|
| Repo skeleton + 3-layer storage + NAS sync | ✅ |
| Conda env (Py 3.11 + torch 2.8 cu128 + nanotron + lighteval 0.13 + datatrove + flash-attn 2/3) | ✅ |
| Reference data (Cosmopedia v2 + FineWeb-Edu) tokenized for both Mistral and Qwen tokenizers | ✅ |
| Dual-protocol architecture (ADR 0007: Llama2-1.82B anchor + Qwen3-1.7B working) | ✅ |
| nanotron training launcher + auto-resume + nohup-resilient | ✅ |
| nanotron Qwen2 → HF Qwen2 converter (ADR 0009) | ✅ |
| lighteval `cosmopedia-full` task group (129 tasks, byte-for-byte parity with Cosmopedia eval.slurm) | ✅ |
| Aim live training-curve monitoring (sidecar — TODO T5 will inline this into nanotron) | ✅ |
| INDEX.parquet experiment registry + NAS cold-mirror | ✅ |
| **R1 · seed42 30B full-scale run** | 🟡 **in progress** (~30h ETA) |
| seed1337 / Track 2 cooldown-fork / Protocol A Llama2 / Incepedia generator / decontam | ⏳ |

Two-weekly progress in [`docs/project-status.md`](./docs/project-status.md).

---

## One-liner

**Multi-generator mixing + web-grounded RAG + multi-agent self-revision + three-layer decontamination, evaluated under a dual-protocol (Llama2-1.82B anchor + Qwen3-1.7B working) Cosmopedia-parity protocol, beats Cosmopedia v1/v2, then scaled to 30B+ for public release.**

In depth:
- [`docs/codenames-cheatsheet.md`](./docs/codenames-cheatsheet.md) — **one-page cheat sheet of all codenames (P/C/T/ADR/Track/Protocol/L/R)**
- [`docs/methodology.md`](./docs/methodology.md) — methodology and decision rationale
- [`docs/decisions/`](./docs/decisions/) — 9 ADRs (architecture, protocol, tooling, eval policy)
- [`docs/incepedia-overview.md`](./docs/incepedia-overview.md) — high-level project narrative
- [`docs/project-status.md`](./docs/project-status.md) — twice-weekly status brief

---

## End-to-end pipeline — train → convert → eval → INDEX → NAS

```
                                    ┌─── methodology.md / ADR 0004 dual-track eval protocol
                                    │
                                    ▼
              experiments/<exp_id>/config.yaml
              (single source of reproducibility — anyone with this yaml can re-run)
                                    │
                                    ▼
            scripts/run_experiment.py (orchestrator)
                                    │
        ┌───────────────────────────┼─────────────────────────────┐
        │                           │                             │
        ▼                           ▼                             ▼
┌───────────────┐         ┌──────────────────┐          ┌─────────────────┐
│ train.log →   │  ←───── │ nanotron train   │ ─────→   │ ckpt/<step>/    │
│ Aim sidecar   │         │ (8×H100 dp,FA2)  │          │  config.yaml    │
│ (live curves) │         │ trapezoidal lr   │          │  model/...      │
└───────────────┘         │ ~55h per 30B/seed│          │  optimizer/...  │
                          │ auto-resume safe │          └─────────────────┘
                          └──────────────────┘                  │
                                                                ▼
                                            scripts/convert_nanotron_qwen2_to_hf.py
                                                          (ADR 0009)
                                                                │
                                                                ▼
                                                ┌──────────────────────────────┐
                                                │ hf_ckpt/                     │
                                                │   config.json (Qwen2ForCausalLM)│
                                                │   model.safetensors (3.5 GB) │
                                                │   tokenizer.json             │
                                                └──────────────────────────────┘
                                                                │
                                                                ▼
                                          lighteval (accelerate, cosmopedia-full)
                                          129 tasks, full samples, ~28 min on 8×H100
                                          (byte-identical to Cosmopedia eval.slurm
                                          → scores directly comparable)
                                                                │
                                                                ▼
                                                ┌────────────────────────┐
                                                │ metrics.json           │
                                                │ INDEX.parquet (+1 row) │
                                                │ NAS sync: config/ckpt/ │
                                                │   hf_ckpt/eval/        │
                                                └────────────────────────┘
```

**Release** (P3 endgame): `hf_ckpt/` → `huggingface_hub.upload_folder` → public HF Hub model repo.

---

## Measured numbers as of now

| Metric | Current (2026-04-23, 19h+ into R1 training) |
|--------|---------------------------------------------|
| Training throughput | 185 K tok/s global / 23 K tok/s/GPU |
| MFU | 28% (BF16 on H100; healthy for 1.7B) |
| Wall clock per 30B/seed | measured ~45-55 h |
| Eval wall clock | 28.5 min (cosmopedia-full 129 tasks, full samples, 8×H100) |
| Convert wall clock | ~30 s (CPU, 150 LOC) |
| Auto-resume | resumes from latest valid ckpt; preview via `scripts/check_resume.py` |

---

## Three-stage roadmap

| Stage | Goal | Done when |
|-------|------|-----------|
| **P1 · Reproduce** | Cosmopedia-parity baseline + dual-protocol × dual-seed | scores match SmolLM/FineWeb published numbers ±0.5pp |
| **P2 · PoC** | Incepedia v0.1 (3B tokens) ≥ Cosmopedia on both tracks | minimal closed loop: OpenRouter generate → tokenize → train → convert → eval |
| **P3 · Iterate & scale** | Incepedia v1.0 (10–30B tokens) decisively beats Cosmopedia | cross-benchmark wins + HF Hub release + paper |

ADR 0004 / 0005 / 0007 for details.

---

## Dual-protocol architecture (ADR 0007)

| Protocol | Architecture | Role | Frequency |
|----------|--------------|------|-----------|
| **A · Llama2-1.82B** | full attention, RoPE θ=10000, no bias, Mistral tokenizer | **External anchor** — match SmolLM/Cosmopedia public numbers ±0.5pp | only **2 runs total** (Cosmopedia v2 × 2 seeds) |
| **B · Qwen3-1.7B** | GQA (8 KV), RoPE θ=1e6, QKV bias, Qwen 151k vocab | **Working architecture** — all Incepedia version ablations + release | 6 milestones × 2 seeds + 33 cooldown-forks |

Invariants: nanotron (native support for both), lighteval cosmopedia-full (same task set), training hyperparams (lr / schedule / seeds identical).

---

## Dual-track eval protocol (ADR 0004)

| Track | Protocol | Question it answers | Use case |
|-------|----------|---------------------|----------|
| **Track 1 · Standalone** | 1.82B/1.7B from scratch × 30B tokens × 2 seeds | "Is dataset X stronger than Y as a standalone pretraining corpus?" | version milestone gating |
| **Track 2 · Seasoning** | shared backbone (FineWeb-Edu × 20B) + cooldown-fork × 6B | "Is dataset X stronger than Y as a decay-stage seasoning?" | day-to-day ablation |

Each milestone runs 2 seeds and averages (ADR 0004: ~±0.15pp noise floor).

---

## Repo layout

```
Incepedia/
├── README.md / README.en.md         # this file
├── .env                              # API keys (gitignored)
├── docs/
│   ├── methodology.md                # methodology in depth
│   ├── project-status.md             # twice-weekly progress brief
│   ├── incepedia-overview.md         # high-level narrative
│   ├── multi-machine-eval-setup.md   # multi-host eval deployment
│   └── decisions/                    # 9 ADRs
├── configs/                          # topics.yaml / personas.yaml / generators.yaml
├── src/incepedia/
│   ├── training/                     # nanotron launcher + Pydantic configs
│   ├── eval/                         # lighteval task defs (cosmopedia-full) + runner
│   ├── generation/                   # OpenRouter async batch generator (P2)
│   └── config.py                     # path constants
├── scripts/
│   ├── run_experiment.py             # train→convert→eval→INDEX→NAS orchestrator
│   ├── convert_nanotron_qwen2_to_hf.py  # ADR 0009
│   ├── check_resume.py               # preview auto-resume state
│   ├── tokenize_dataset.py           # datatrove parquet → nanotron .ds
│   ├── prefetch_eval_datasets.py     # offline-cache eval data (avoid HF 429)
│   ├── lint_nanotron_yaml.py         # audit launcher YAML vs nanotron defaults
│   ├── tail_train_log_to_aim.py      # train.log → Aim sidecar (interim; T5 replaces)
│   ├── bench_fa2_vs_fa3.py           # FA2 vs FA3 numerical / throughput benchmark
│   ├── stats_cosmopedia_v2.py        # upstream data format/audience/token_length stats
│   ├── sync_to_nas.sh                # event-driven rsync (ckpt/hf_ckpt/gen/eval/...)
│   ├── mount_lambda_nas.sh           # NAS mount on the A100 box (virtiofs/sshfs/rsync)
│   └── bootstrap_env.sh              # one-shot conda env + apply patches
├── patches/                          # 4 local nanotron patches (rotary/datatrove/cache/FA3)
│   └── nanotron_*.patch              # bootstrap idempotent-applies them
├── data/                             # bulk data (gitignored)
│   ├── datasets/                     # tokenized .ds shards (training hot-path)
│   ├── reference/                    # Cosmopedia v2 / FineWeb-Edu raw parquet
│   ├── hf_cache/                     # isolated eval-dataset cache
│   └── raw_generations/              # synthesized batch parquet (P2)
├── experiments/
│   ├── exp_ref_cosmopedia_v2_qwen3_seed{42,1337}/   # Protocol B baseline
│   ├── exp_ref_cosmopedia_v2_seed{42,1337}/          # Protocol A baseline
│   ├── backbone_fineweb_edu_qwen3/                   # Track 2 shared backbone
│   └── exp_<exp_id>/
│       ├── config.yaml               # self-contained, single reproducibility source
│       ├── nanotron.yaml             # launcher-rendered (gitignored)
│       ├── ckpt/<step>/              # nanotron native ckpts (gitignored, NAS)
│       ├── hf_ckpt/                  # HF Qwen2 converted (gitignored, NAS)
│       ├── train.log                 # tee'd nanotron stdout (gitignored)
│       ├── eval/                     # full lighteval results
│       ├── metrics.json              # 14-16 benchmark scores (in git)
│       └── README.md                 # human-readable summary (optional)
├── aim/                              # Aim runtime DB (gitignored)
└── INDEX.parquet                     # all-experiments roll-up (in git)
```

---

## Three-layer storage defense (ADR 0001)

| Layer | Location | Contents | Copies |
|-------|----------|----------|--------|
| **L1 · Hot** | local NVMe `/home/ubuntu/lilei/projects/Incepedia/` | full hot+cold, single source of truth for train/eval/gen | 1 |
| **L2 · Cold mirror** | NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/` (virtiofs) | event-driven rsync; 3.2 PB | +1 |
| **L3 · Metadata fallback** | git remote (GitHub) | configs/ + experiments/*/config.yaml + metrics.json + INDEX.parquet (<100 MB) | +1 |

**Worst case**: local SSD + NAS lost simultaneously → git restores all configs + eval verdicts; only weights lost (re-trainable from same config).

---

## Sync strategy (event-driven)

| Trigger | Target | Sub-command |
|---------|--------|-------------|
| Checkpoint save complete | `experiments/<exp>/ckpt/<step>/` | `sync_to_nas.sh ckpt <exp_id>` |
| HF conversion done (auto) | `experiments/<exp>/hf_ckpt/` | `sync_to_nas.sh hf_ckpt <exp_id>` |
| Generation batch complete | `data/raw_generations/<batch>/` | `sync_to_nas.sh gen <batch_id>` |
| Experiment launch | `config.yaml` (immediate) | `sync_to_nas.sh config <exp_id>` |
| Eval complete | `experiments/<exp>/eval/` | `sync_to_nas.sh eval <exp_id>` |
| Tokenized dataset landed | `data/datasets/<id>/` | `sync_to_nas.sh dataset <id>` |
| Nightly 03:00 | full safety net | cron (TBD) |

---

## Multi-machine deployment — 165 (8×H100 train) + 164 (8×A100 eval)

> **Architecture**: 165 trains, 164 evaluates; both share ckpt/eval/INDEX through NAS.  
> **Status**: 165 live; 164 awaits NAS attach (virtiofs needs Lambda console attach OR sshfs fallback). See [`docs/multi-machine-eval-setup.md`](./docs/multi-machine-eval-setup.md).

```
H100 (8×80G, host 165)        NAS (3.2 PB)         A100 (8×40G, host 164)
─────────────────────       ─────────────────       ─────────────────
nanotron training           experiments/             lighteval eval
       ↓ convert            ├─ <exp>/ckpt/           ↑ rsync HF ckpt
       ↓ sync               ├─ <exp>/hf_ckpt/        ↓ write metrics
                            ├─ <exp>/eval/           ↑ rsync back
                            └─ INDEX.parquet
                              (single source of truth)
```

---

## Quick start

### One-shot environment setup

```bash
cd /home/ubuntu/lilei/projects/Incepedia
bash scripts/bootstrap_env.sh        # conda env + nanotron + datatrove + 4 local patches
```

### Run a reference experiment (end-to-end)

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate incepedia

# Pre-flight: GPU availability
nvidia-smi --query-compute-apps=pid --format=csv,noheader

# Pre-flight: auto-resume preview
python scripts/check_resume.py experiments/exp_ref_cosmopedia_v2_qwen3_seed42/config.yaml

# Launch (nohup-resilient; INCEPEDIA_FRESH_START=1 forces clean start)
export CUDA_DEVICE_MAX_CONNECTIONS=1
nohup python scripts/run_experiment.py \
    --config experiments/exp_ref_cosmopedia_v2_qwen3_seed42/config.yaml \
    > /tmp/seed42_train.out 2>&1 &

# Monitor
tail -F experiments/exp_ref_cosmopedia_v2_qwen3_seed42/train.log | grep iteration:
# Aim Web UI (assumes SSH tunnel)
open http://localhost:43800/
```

### Eval an existing checkpoint only

```bash
python scripts/run_experiment.py \
    --config experiments/<exp>/config.yaml --eval-only
# auto-converts if needed, runs lighteval cosmopedia-full → metrics.json → INDEX → NAS
```

### Resume a crashed training

```bash
# just re-run the same command; launcher auto-detects latest valid ckpt
nohup python scripts/run_experiment.py --config <...>/config.yaml > /tmp/resume.out 2>&1 &
```

---

## API key configuration

```bash
# dual location (global + repo); repo wins; loaded via python-dotenv
# global:  ~/.bash_profile  →  export OPENROUTER_API_KEY=sk-or-v1-...
# repo  :  .env             →  (gitignored)
grep OPENROUTER_API_KEY .env && echo OK
```

Code lookup order: `OPENROUTER_API_KEY` → `OpenRouter_API_KEY` → `OPENAI_API_KEY` (compat).

---

## Experiment naming convention

```
exp_{semantic_slug}_{seed?}
```

Examples:
- `exp_ref_cosmopedia_v2_qwen3_seed42` — Protocol B Cosmopedia v2 baseline
- `exp_ref_cosmopedia_v2_seed1337` — Protocol A Cosmopedia v2 baseline second seed
- `exp_inc_v01_qwen3_seed42` — Incepedia v0.1 first run on Protocol B (P2)
- `backbone_fineweb_edu_qwen3` — Track 2 shared backbone

Each experiment dir must contain `config.yaml` (single reproducibility source) + `track: 1|2` field (ADR 0004).

---

## Reading order for agents / new contributors

| # | Document | Purpose |
|---|----------|---------|
| 1 | this README | 5-minute project overview |
| 2 | [`docs/project-status.md`](./docs/project-status.md) | current state, TODO queue, risks |
| 3 | [`docs/methodology.md`](./docs/methodology.md) | full methodology, dual-track eval, Incepedia generation strategy (C1-C16) |
| 4 | [`docs/incepedia-overview.md`](./docs/incepedia-overview.md) | high-level narrative |
| 5 | [`docs/decisions/README.md`](./docs/decisions/README.md) | 9-ADR index |
| 6 | `INDEX.parquet` (`pandas.read_parquet`) | latest experiment status + benchmark scores |
| 7 | most recently modified `experiments/exp_*/` | what's actively running |

Agent rules in [`AGENTS.md`](./AGENTS.md); contribution guide in [`CONTRIBUTING.md`](./CONTRIBUTING.md).

---

## Key references

- [Cosmopedia blog](https://huggingface.co/blog/cosmopedia)
- [SmolLM blog (v2 recipe)](https://huggingface.co/blog/smollm)
- [FineWeb paper (ablation paradigm)](https://arxiv.org/abs/2406.17557)
- [Cosmopedia GitHub](https://github.com/huggingface/cosmopedia) (vendored at `third_party/cosmopedia/`)
- [Phi-4 technical report](https://arxiv.org/abs/2412.08905)
- [Hägele et al. 2024 cooldown-fork](https://arxiv.org/abs/2405.18392) (Track 2 protocol basis)

---

© 2026 · Incepedia contributors · Apache-2.0
