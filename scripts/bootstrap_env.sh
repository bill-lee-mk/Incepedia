#!/usr/bin/env bash
# Incepedia · conda 环境一键搭建
# 用法:
#   bash scripts/bootstrap_env.sh           # 默认环境名 incepedia
#   ENV_NAME=incepedia310 bash scripts/bootstrap_env.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-incepedia}"
PY_VER="${PY_VER:-3.10}"
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

# --- HF stack (source installs, pinned commits) ---
# These change often; we install main-branch builds.
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
clone_or_pull https://github.com/huggingface/lighteval.git    "$THIRD_PARTY_DIR/lighteval"
clone_or_pull https://github.com/huggingface/datatrove.git    "$THIRD_PARTY_DIR/datatrove"
clone_or_pull https://github.com/huggingface/cosmopedia.git   "$THIRD_PARTY_DIR/cosmopedia"

pip install -e "$THIRD_PARTY_DIR/nanotron"  || echo "[warn] nanotron install failed — will retry later"
pip install -e "$THIRD_PARTY_DIR/lighteval[accelerate,quantization,adapters]" \
  || echo "[warn] lighteval install failed — will retry later"
pip install -e "$THIRD_PARTY_DIR/datatrove"  || echo "[warn] datatrove install failed — will retry later"

# --- torch (CUDA 12.x for H100) ---
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo
echo "[done] To activate:    conda activate $ENV_NAME"
echo "[done] To verify:      bash scripts/check_setup.sh"
