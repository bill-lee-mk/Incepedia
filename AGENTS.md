# AGENTS.md — Guidelines for AI Agents working on Incepedia

This document is the **primary instruction set** for any AI coding agent (Cursor, Claude Code, Codex, etc.) operating on this repository. Read it **before** making any change.

---

## Project context

Incepedia is a synthetic pretraining dataset project aiming to **outperform Cosmopedia v2** on the standard 1.82B Llama2 / 30B-token ablation protocol. See `README.md` for the full picture.

The shared server hosting this repo is used by **multiple contributors** — do not assume global state (git identity, shell env, credentials, installed packages) belongs to the current user.

---

## Golden rules

### 1. Never commit with the wrong identity
- This repo pins a **local** `user.name` / `user.email` via `.git/config`. Do NOT override it.
- Before any commit run `git config --local user.name` and verify it equals `bill-lee-mk` (or whoever the current owner is).
- NEVER touch `~/.gitconfig` global settings.

### 2. Never leak secrets to stdout
- `.env` contains tokens. Do NOT `cat .env`, `echo $VAR` for tokens, or paste tokens into chat/log output, even masked.
- When reading credentials: use Python `dotenv` / `os.environ`. When writing credential files: `printf` into the file directly without echoing to terminal.
- If a secret does leak, tell the user **immediately** so they can rotate.

### 3. Never push without user authorization for high-risk ops
- Routine `git commit` + `git push` on feature/main additions is OK and should be **automatic** after each meaningful change.
- For `git push --force`, `git reset --hard` on shared branches, deletions, or history rewrites: **stop and ask**.

### 4. Storage boundary is sacred
- **Local NVMe** = source of truth, all hot/cold data lives here
- **NAS** (`/lambda/nfs/us-south-2/incepedia_exp_bak/`) = cold mirror only, no training/eval runs here
- **git remote** = metadata fallback for `<1GB` structured artifacts only

Respect the three-layer defense. Never train/eval directly on NAS; never commit `data/` or `ckpt/` to git.

### 5. Use the established tooling
- Experiment tracking: **Aim** (local, in `aim/`), not WandB
- Dedup/tokenization: **datatrove**
- Training: **nanotron**
- Evaluation: **lighteval** with the exact task definitions ported from `cosmopedia/evaluation/`
- Generation: **OpenRouter** via async HTTP client in `src/incepedia/generation/`
- Never introduce a new framework to solve a problem the listed tools already solve.

### 6. Ablation discipline
- Every experiment changes **exactly one variable** relative to its parent. Record the parent in `config.yaml:branch_from`.
- Run 2 seeds and average before claiming any result.
- Only compare within the same training-token budget and model config.

---

## Auto-commit workflow (standard for agents)

After completing a meaningful unit of change:

1. `git status` to see what's touched
2. Stage only intended files — **never** `git add .` blindly (could add data/ckpts if rules slipped)
3. Verify no secrets, no large binaries:
   ```bash
   git diff --cached --stat
   git diff --cached | grep -iE 'sk-or-|ghp_|hf_[A-Za-z0-9]{20,}' && { echo "SECRET DETECTED — abort"; exit 1; }
   ```
4. Compose a Conventional-Commits style message:
   ```
   <type>(<scope>): <imperative summary ≤72 chars>

   <body: why, not what, ≤5 bullets>
   ```
   Types: `feat` / `fix` / `docs` / `chore` / `refactor` / `exp` / `data` / `eval`.
5. `git commit` with the local identity.
6. `git push origin main` (or the current branch).
7. Report in chat: commit hash + one-line summary + remote URL.

---

## Repository conventions

### Directory → allowed content

| Directory | Tracked? | Content |
|---|---|---|
| `src/incepedia/` | ✅ | Python package code |
| `scripts/` | ✅ | Shell / Python entrypoints |
| `configs/` | ✅ | YAML/Jinja prompts, topic/persona tables |
| `docs/` | ✅ | Methodology, FAQ, ADRs |
| `experiments/*/config.yaml` | ✅ | Experiment spec (required, self-contained) |
| `experiments/*/metrics.json` | ✅ | Eval scores (required after run) |
| `experiments/*/README.md` | ✅ | Plain-English summary (required) |
| `experiments/*/ckpt/` | ❌ | Model weights → NAS only |
| `experiments/*/eval/` | ❌ | Raw lighteval outputs → NAS only |
| `data/` | ❌ | All datasets, generations, tokenizers |
| `aim/` | ❌ | Aim runtime DB |
| `logs/` | ❌ | Training stdout |
| `.env` | ❌ | Secrets (use `.env.example` as template) |

### Experiment naming

```
exp_{YYYYMMDD}_{variant_slug}_seed{N}
```

Always lowercase, no spaces, hyphens OK in slug. Examples:
- `exp_20260420_reference_cosmopedia_v2_seed42`
- `exp_20260425_incepedia_v0.1_multigen_seed1337`

### Config file format (`experiments/*/config.yaml`)

Must contain at minimum:
```yaml
exp_id: ...
stage: P1 | P2 | P3
branch_from: null | <parent_exp_id>:<ckpt_step>
dataset:
  id: ...
  path: ...
  tokens: ...
model:
  arch: llama2-1.82B
  seq_len: 2048
  tokenizer: ...
training:
  global_batch_tokens: 1310720
  lr: 3.0e-4
  scheduler: trapezoidal | cosine
  cooldown_fraction: 0.20
  seed: 42
eval:
  tasks: early-signal
notes: "one-line plain-English"
```

---

## When in doubt, ASK
Prefer asking the user over guessing on:
- Credential handling
- Deleting anything from NAS
- Changing tracked branch policy
- Introducing new dependencies not in `requirements.txt`
- Anything that could cost >$100 in API spend without prior agreement

---

© 2026 · Incepedia contributors
