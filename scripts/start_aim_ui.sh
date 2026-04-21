#!/usr/bin/env bash
# Start the Aim Web UI for browsing training curves.
#
# Usage:
#   bash scripts/start_aim_ui.sh              # default port 43800
#   AIM_PORT=8000 bash scripts/start_aim_ui.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIM_DIR="${AIM_DIR:-$REPO_ROOT/aim}"
AIM_PORT="${AIM_PORT:-43800}"
AIM_HOST="${AIM_HOST:-0.0.0.0}"

mkdir -p "$AIM_DIR"

if [[ ! -f "$AIM_DIR/.aim/meta" ]]; then
  echo "[aim] initialising repo at $AIM_DIR"
  (cd "$AIM_DIR" && aim init)
fi

echo "[aim] serving at http://$AIM_HOST:$AIM_PORT  (repo: $AIM_DIR)"
exec aim up --host "$AIM_HOST" --port "$AIM_PORT" --repo "$AIM_DIR"
