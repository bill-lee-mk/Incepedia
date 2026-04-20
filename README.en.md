# Incepedia

> [🇨🇳 中文](./README.md) · 🇬🇧 English (current) · [📦 NAS cold-mirror](/lambda/nfs/us-south-2/incepedia_exp_bak/README.en.md)

A synthetic pretraining dataset project built on Cosmopedia's architecture, aiming to **outperform Cosmopedia** on downstream benchmarks.

---

## One-liner

**Multi-generator mixing + web-grounded RAG + multi-agent self-revision + three-layer decontamination, benchmarked against Cosmopedia v2 on the 1.82B Llama2 / 30B-token ablation protocol, then scaled to 10B+.**

## Current stage

🟡 **Stage 0 · Scaffolding** (this README is the deliverable of this stage)

- [x] OpenRouter API key ready
- [x] Storage architecture finalized (local full · NAS cold mirror · git metadata fallback)
- [x] NAS benchmark (virtiofs, 885 MB/s write, 3.4 GB/s parallel)
- [ ] Repo directory skeleton
- [ ] Dependency environment (nanotron / lighteval / datatrove)
- [ ] Reference baseline: reproduce Cosmopedia v2 @ 30B tokens
- [ ] OpenRouter async batch generator
- [ ] `INDEX.parquet` experiment registry + sync hooks

## Design philosophy (short form)

1. **Data quality == downstream model score.** No subjective judgments — everything proved through ablation training.
2. **"Web-grounded synthetic" > generation from scratch.** Every generation is seed-grounded to reduce hallucination.
3. **Route bottlenecks per task**: mid-tier models for common-sense bulk; frontier models for math / reasoning / code.
4. **Diversity is a first-class citizen.** Multi-generator × persona × difficulty × structure × embedding-level dedup.
5. **Three-layer decontamination**: n-gram + embedding similarity + LLM-as-judge.

Full methodology in [`docs/methodology.md`](./docs/methodology.md) (TBD).

## Directory layout

```
Incepedia/
├── README.md / README.en.md         # This file
├── .env                              # API keys (in .gitignore)
├── docs/                             # methodology, experiment notes, ADRs
├── configs/                          # topics.yaml / personas.yaml / prompt_templates/
├── src/
│   ├── generation/                   # OpenRouter async batch generator
│   ├── dedup/                        # MinHash + embedding dedup
│   ├── decontam/                     # Three-layer decontamination
│   ├── training/                     # nanotron training entry
│   └── eval/                         # lighteval task defs (ported from cosmopedia)
├── scripts/
│   ├── sync_to_nas.sh                # Event-driven rsync
│   ├── index_experiment.py           # INDEX.parquet maintainer
│   └── ...
├── data/                             # Large data (in .gitignore)
│   ├── datasets/                     # Tokenized shards (training hot-path)
│   ├── raw_generations/              # Raw synthetic Parquet
│   ├── reference/                    # Cosmopedia v2 / FineWeb-Edu copies
│   └── tokenizers/
├── experiments/                      # All ablations (ckpt/ subdir ignored, rest tracked)
│   └── exp_{YYYYMMDD}_{slug}_{variant}/
│       ├── config.yaml               # Fully self-contained; reproducible from this alone
│       ├── metrics.json              # Training loss / eval scores per 2B tokens
│       ├── eval/                     # Full lighteval results
│       ├── ckpt/                     # Model snapshots (not in git, synced to NAS)
│       └── README.md                 # Plain-English summary: what changed, what learned
├── aim/                              # Aim runtime DB (in .gitignore)
└── INDEX.parquet                     # Aggregate view of all experiments (in git)
```

## Three-layer storage defense

| Layer | Location | Content | Copies |
|---|---|---|---|
| L1 · Hot | Local NVMe `/home/ubuntu/lilei/projects/Incepedia/` | Full hot + cold; single source of truth | 1 |
| L2 · Cold mirror | NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/` | Event-driven rsync cold copy; entry point for team / agents | +1 (cold) |
| L3 · Metadata fallback | git remote | `configs/`, `experiments/*/config.yaml`, `metrics.json`, `INDEX.parquet`, all <1GB structured | +1 (meta) |

**Worst case**: both local SSD and NAS die → can still recover all experiment configs and eval conclusions from git; only model weights are lost, retrainable from the same configs.

## Sync strategy (event-driven)

| Trigger | Target | Script |
|---|---|---|
| Checkpoint saved | `experiments/exp_xxx/ckpt/step_N/` | `scripts/sync_to_nas.sh ckpt` |
| Generation batch done | `data/raw_generations/batch_xxx/` | `scripts/sync_to_nas.sh gen` |
| Experiment launched | `config.yaml` (immediate) | `scripts/sync_to_nas.sh config` |
| Eval finished | `eval/` dir | `scripts/sync_to_nas.sh eval` |
| Every 03:00 | Full safety-net rsync | cron |

**Exclusions**: `aim/` / `logs/` / `.env` / `.git` / `*.tmp` / `*.lock`.

## Quick start

```bash
# 1. Enter repo
cd /home/ubuntu/lilei/projects/Incepedia

# 2. Verify .env (key configured)
grep OPENROUTER_API_KEY .env

# 3. Create env (requirements.txt to be added)
conda create -n incepedia python=3.10 -y
conda activate incepedia
pip install -r requirements.txt   # TBD

# 4. Verify NAS writable
touch /lambda/nfs/us-south-2/incepedia_exp_bak/.probe && rm /lambda/nfs/us-south-2/incepedia_exp_bak/.probe && echo "NAS OK"

# 5. Run first reference experiment (TBD)
python scripts/run_experiment.py --config experiments/exp_001_reference_cosmopedia_v2/config.yaml
```

## API key setup

Two-tier config (global + repo-level). **Repo-level takes priority**; code loads via `python-dotenv`.

- **Global**: `~/.bash_profile` → `export OpenRouter_API_KEY=sk-or-v1-...`
- **Repo-level**: `.env` at repo root (already in `.gitignore`)

Code resolves in order: `OPENROUTER_API_KEY` → `OpenRouter_API_KEY` → `OPENAI_API_KEY` (compatibility shim in place).

## Experiment naming convention

```
exp_{YYYYMMDD}_{variant_slug}_{seed?}
```
Examples:
- `exp_20260420_reference_cosmopedia_v2_seed42`
- `exp_20260425_incepedia_v0.1_multigen_seed1337`

## Three-phase roadmap

| Phase | Goal | Deliverable |
|---|---|---|
| **P1 · Reproduce** | Match Cosmopedia v2 early-signal scores within ±0.5pp on 1.82B/30B-token ablation | reference baseline + training/eval pipeline |
| **P2 · PoC** | 3B-token Incepedia v0.1 matches or beats Cosmopedia v2 at same setup | Minimal closed loop: generate → train → eval → compare |
| **P3 · Iterate & scale** | Use ablation feedback to scale to 10B/15B/30B, beat Cosmopedia v2 across benchmarks, approach SmolLM2 | Incepedia v1.0 release candidate |

## Key references

- [Cosmopedia blog](https://huggingface.co/blog/cosmopedia)
- [SmolLM blog (v2 recipe)](https://huggingface.co/blog/smollm)
- [FineWeb paper (ablation protocol)](https://arxiv.org/abs/2406.17557)
- [Cosmopedia GitHub](https://github.com/huggingface/cosmopedia)
- [Phi-4 technical report](https://arxiv.org/abs/2412.08905)

## For agents / new contributors

After reading this README, proceed in order:
1. [`docs/methodology.md`](./docs/methodology.md) — methodology and decision rationale (TBD)
2. [`docs/decisions/`](./docs/decisions/) — ADRs (TBD)
3. `INDEX.parquet` — latest experiment status and best scores
4. Recently modified `experiments/exp_*/README.md` — understand work in progress

If unclear, read `docs/FAQ.md` (TBD) first, then file an issue.

---

© 2026 · Incepedia contributors · Apache-2.0
