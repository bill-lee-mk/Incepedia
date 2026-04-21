#!/usr/bin/env bash
# Incepedia · conda 环境一键搭建
# 用法:
#   bash scripts/bootstrap_env.sh           # 默认环境名 incepedia
#   ENV_NAME=incepedia310 bash scripts/bootstrap_env.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-incepedia}"
PY_VER="${PY_VER:-3.11}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "────────────────────────────────────────────────────────"
echo "Incepedia bootstrap"
echo "  repo  : $REPO_ROOT"
echo "  env   : $ENV_NAME  (python $PY_VER)"
echo "────────────────────────────────────────────────────────"

# --- conda sanity (auto-source from common locations if not on PATH) ---
if ! command -v conda &>/dev/null; then
  for candidate in \
      /opt/conda /opt/miniforge3 /opt/miniconda3 \
      "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "$candidate/etc/profile.d/conda.sh"
      echo "[info] sourced conda from $candidate"
      break
    fi
  done
fi
if ! command -v conda &>/dev/null; then
  echo "[ERR] conda not found. Install miniforge / miniconda first."
  exit 1
fi

# --- create env if missing ---
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[info] conda env '$ENV_NAME' exists, reusing."
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- base deps ---
pip install --upgrade pip wheel setuptools
pip install -r "$REPO_ROOT/requirements.txt"

# --- editable install of incepedia package ---
pip install -e "$REPO_ROOT"

# --- torch with CUDA 12.8 (H100 / Hopper; matches driver 570.x) ---
# We pin torch==2.8.0 because:
#   (1) it is the lowest torch release with official cu128 wheels,
#   (2) lighteval 0.13 supports torch<3.0,
#   (3) flash_attn_3 stable wheels target cu128+torch2.8 (widest coverage).
pip install --upgrade "torch==2.8.0" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# --- FlashAttention 3 (stable) — Hopper-optimized, ~2x faster than torch SDPA on H100 ---
# Uses pre-built wheels (requires sm_90 / H100). Pip package: `flash_attn_3`.
# Import path in Python: `from flash_attn_interface import flash_attn_func`
# In transformers: `AutoModel.from_pretrained(..., attn_implementation="flash_attention_3")`
# Does NOT conflict with legacy `flash-attn` (FA2). See docs/decisions/0007-flash-attn-3.md.
pip install --upgrade "flash_attn_3>=3.0.0,<3.1" \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 \
  --only-binary flash_attn_3

# --- lighteval from pypi (pinned; see docs/decisions/0006-evaluation-stack-policy.md) ---
pip install "lighteval==0.13.0"

# --- nanotron + datatrove from source (cosmopedia read-only for reference) ---
THIRD_PARTY_DIR="$REPO_ROOT/third_party"
mkdir -p "$THIRD_PARTY_DIR"

clone_or_pull () {
  local url="$1" dir="$2"
  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" pull --ff-only || true
  else
    git clone --depth 1 "$url" "$dir"
  fi
}

clone_or_pull https://github.com/huggingface/nanotron.git     "$THIRD_PARTY_DIR/nanotron"
clone_or_pull https://github.com/huggingface/datatrove.git    "$THIRD_PARTY_DIR/datatrove"
clone_or_pull https://github.com/huggingface/cosmopedia.git   "$THIRD_PARTY_DIR/cosmopedia"

pip install -e "$THIRD_PARTY_DIR/nanotron"  || echo "[warn] nanotron install failed — retry manually"
pip install -e "$THIRD_PARTY_DIR/datatrove" || echo "[warn] datatrove install failed — retry manually"

echo
echo "[done] To activate:    conda activate $ENV_NAME"
echo "[done] To verify:      bash scripts/check_setup.sh"
