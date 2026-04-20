"""Incepedia · runtime config.

- loads `.env` from repo root (via python-dotenv) if present
- exposes canonical paths for local vs NAS
- provides `resolve_openrouter_key()` that accepts any of:
    OPENROUTER_API_KEY  (preferred)
    OpenRouter_API_KEY  (legacy camelCase)
    OPENAI_API_KEY      (back-compat when base_url points to openrouter)
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)

# ─── Canonical paths ──────────────────────────────────────────────────
LOCAL_ROOT: Path = Path(os.environ.get("INCEPEDIA_LOCAL_ROOT", str(REPO_ROOT))).resolve()
NAS_ROOT: Path = Path(
    os.environ.get("INCEPEDIA_NAS_ROOT", "/lambda/nfs/us-south-2/incepedia_exp_bak")
).resolve()

DATA_DIR: Path = LOCAL_ROOT / "data"
DATASETS_DIR: Path = DATA_DIR / "datasets"
RAW_GEN_DIR: Path = DATA_DIR / "raw_generations"
REFERENCE_DIR: Path = DATA_DIR / "reference"
TOKENIZERS_DIR: Path = DATA_DIR / "tokenizers"
EXPERIMENTS_DIR: Path = LOCAL_ROOT / "experiments"
CONFIGS_DIR: Path = LOCAL_ROOT / "configs"
DOCS_DIR: Path = LOCAL_ROOT / "docs"
LOGS_DIR: Path = LOCAL_ROOT / "logs"
AIM_DIR: Path = LOCAL_ROOT / "aim"
INDEX_PARQUET: Path = LOCAL_ROOT / "INDEX.parquet"

# ─── OpenRouter ───────────────────────────────────────────────────────
OPENROUTER_BASE_URL: str = os.environ.get(
    "OPENAI_API_BASE", "https://openrouter.ai/api/v1"
)


def resolve_openrouter_key() -> str:
    """Return a usable OpenRouter key, checking multiple env-var conventions."""
    for name in ("OPENROUTER_API_KEY", "OpenRouter_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(name)
        if val and val.startswith("sk-or-"):
            return val
    raise RuntimeError(
        "OpenRouter API key not found in env. Set OPENROUTER_API_KEY in .env or shell."
    )


def resolve_second_openrouter_key() -> str | None:
    """Optional second key for concurrency / rotation."""
    val = os.environ.get("OPENROUTER_API_KEY_2")
    if val and val.startswith("sk-or-"):
        return val
    return None


# ─── Ensure dirs (idempotent, lazy) ───────────────────────────────────
def ensure_dirs() -> None:
    for p in (DATA_DIR, DATASETS_DIR, RAW_GEN_DIR, REFERENCE_DIR, TOKENIZERS_DIR,
              EXPERIMENTS_DIR, CONFIGS_DIR, LOGS_DIR, AIM_DIR):
        p.mkdir(parents=True, exist_ok=True)
