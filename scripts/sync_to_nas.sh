#!/usr/bin/env bash
# Incepedia · 事件驱动 rsync → NAS
#
# 用法:
#   sync_to_nas.sh ckpt    <exp_id>             # 同步单个实验 nanotron 训练 ckpt
#   sync_to_nas.sh hf_ckpt <exp_id>             # 同步 HF 转换格式 ckpt(eval/发布用)
#   sync_to_nas.sh gen     <batch_id>           # 同步一轮生成批次
#   sync_to_nas.sh config  <exp_id>             # 立即同步 config.yaml
#   sync_to_nas.sh eval    <exp_id>             # 同步评测结果
#   sync_to_nas.sh dataset <dataset_id>         # 同步 tokenized 数据集
#   sync_to_nas.sh reference <ref_id>           # 同步 data/reference/<ref_id>/(原始参考数据)
#   sync_to_nas.sh nightly                      # 兜底全量(cron 用)
#   sync_to_nas.sh dry     <type> <args...>     # 所有 type 前加 dry 都是 --dry-run
#
# 安全:排除 .env / .git / aim/ / logs/ / *.tmp / *.lock / credentials
# 审计:每次成功同步 append 一行到 NAS 上的 _sync_manifest.jsonl

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAS_ROOT="${INCEPEDIA_NAS_ROOT:-/lambda/nfs/us-south-2/incepedia_exp_bak}"
HOSTNAME_SHORT="$(hostname -s)"

COMMON_EXCLUDES=(
  --exclude='.env' --exclude='.env.*' --exclude='.git/'
  --exclude='aim/' --exclude='logs/' --exclude='.venv/' --exclude='venv/'
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.tmp' --exclude='*.lock'
  --exclude='.sync_in_progress' --exclude='.credentials*'
)

DRY=""
if [[ "${1:-}" == "dry" ]]; then
  DRY="--dry-run"
  shift
fi

MODE="${1:-}"
shift || true

ensure_nas () {
  [[ -d "$NAS_ROOT" ]] || { echo "[ERR] NAS not mounted: $NAS_ROOT"; exit 1; }
  mkdir -p "$NAS_ROOT"
}

audit_log () {
  # $1 type  $2 target  $3 bytes  $4 files
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local line
  line=$(printf '{"ts":"%s","host":"%s","type":"%s","target":"%s","bytes":%s,"files":%s}' \
               "$ts" "$HOSTNAME_SHORT" "$1" "$2" "$3" "$4")
  if [[ -z "$DRY" ]]; then
    echo "$line" >> "$NAS_ROOT/_sync_manifest.jsonl"
  fi
  echo "[audit] $line"
}

rsync_with_audit () {
  # $1 type  $2 src  $3 dst
  local type="$1" src="$2" dst="$3"
  mkdir -p "$(dirname "$dst")"
  local tmplog
  tmplog=$(mktemp)
  # shellcheck disable=SC2086
  rsync -a --human-readable --stats $DRY "${COMMON_EXCLUDES[@]}" "$src" "$dst" > "$tmplog"
  local bytes files
  bytes=$(awk '/Total transferred file size/ {gsub(",","",$5); print $5}' "$tmplog" | head -1)
  files=$(awk '/Number of regular files transferred/ {print $NF}' "$tmplog" | head -1)
  bytes=${bytes:-0}
  files=${files:-0}
  rm -f "$tmplog"
  audit_log "$type" "$dst" "$bytes" "$files"
}

case "$MODE" in
  ckpt)
    ensure_nas
    EXP_ID="${1:?需要 exp_id}"
    SRC="$REPO_ROOT/experiments/$EXP_ID/ckpt/"
    DST="$NAS_ROOT/experiments/$EXP_ID/ckpt/"
    [[ -d "$SRC" ]] || { echo "[ERR] no ckpt dir: $SRC"; exit 1; }
    rsync_with_audit "ckpt" "$SRC" "$DST"
    ;;

  hf_ckpt)
    # Sync the converted (HF transformers format) checkpoint to NAS.
    # This is the artifact eval / publishing flows actually consume — much
    # smaller than the raw nanotron ckpt (~3.5 GB vs ~28 GB for our 1.7B).
    ensure_nas
    EXP_ID="${1:?需要 exp_id}"
    SRC="$REPO_ROOT/experiments/$EXP_ID/hf_ckpt/"
    DST="$NAS_ROOT/experiments/$EXP_ID/hf_ckpt/"
    [[ -d "$SRC" ]] || { echo "[ERR] no hf_ckpt dir: $SRC (run scripts/convert_nanotron_qwen2_to_hf.py first)"; exit 1; }
    rsync_with_audit "hf_ckpt" "$SRC" "$DST"
    ;;

  gen)
    ensure_nas
    BATCH_ID="${1:?需要 batch_id}"
    SRC="$REPO_ROOT/data/raw_generations/$BATCH_ID/"
    DST="$NAS_ROOT/data/raw_generations/$BATCH_ID/"
    [[ -d "$SRC" ]] || { echo "[ERR] no batch dir: $SRC"; exit 1; }
    rsync_with_audit "gen" "$SRC" "$DST"
    ;;

  config)
    ensure_nas
    EXP_ID="${1:?需要 exp_id}"
    SRC="$REPO_ROOT/experiments/$EXP_ID/"
    DST="$NAS_ROOT/experiments/$EXP_ID/"
    # only config.yaml + README + metrics (no ckpt/eval here, they have own syncs)
    mkdir -p "$DST"
    for f in config.yaml README.md metrics.json; do
      [[ -f "$SRC/$f" ]] && rsync -a $DRY "$SRC/$f" "$DST/$f"
    done
    audit_log "config" "$DST" 0 3
    ;;

  eval)
    ensure_nas
    EXP_ID="${1:?需要 exp_id}"
    SRC="$REPO_ROOT/experiments/$EXP_ID/eval/"
    DST="$NAS_ROOT/experiments/$EXP_ID/eval/"
    [[ -d "$SRC" ]] || { echo "[ERR] no eval dir: $SRC"; exit 1; }
    rsync_with_audit "eval" "$SRC" "$DST"
    ;;

  dataset)
    ensure_nas
    DS_ID="${1:?需要 dataset_id}"
    SRC="$REPO_ROOT/data/datasets/$DS_ID/"
    DST="$NAS_ROOT/data/datasets/$DS_ID/"
    [[ -d "$SRC" ]] || { echo "[ERR] no dataset dir: $SRC"; exit 1; }
    rsync_with_audit "dataset" "$SRC" "$DST"
    ;;

  reference)
    ensure_nas
    REF_ID="${1:?需要 reference id (e.g. cosmopedia_v2)}"
    SRC="$REPO_ROOT/data/reference/$REF_ID/"
    DST="$NAS_ROOT/data/reference/$REF_ID/"
    [[ -d "$SRC" ]] || { echo "[ERR] no reference dir: $SRC"; exit 1; }
    rsync_with_audit "reference" "$SRC" "$DST"
    ;;

  nightly)
    ensure_nas
    # Safety-net full sync. Strict excludes of big-but-replaceable runtime dirs.
    rsync -a --human-readable $DRY "${COMMON_EXCLUDES[@]}" \
      --exclude='experiments/*/eval/raw_*'  \
      "$REPO_ROOT/configs/"      "$NAS_ROOT/configs/"
    rsync -a --human-readable $DRY "${COMMON_EXCLUDES[@]}" \
      "$REPO_ROOT/experiments/"  "$NAS_ROOT/experiments/"
    rsync -a --human-readable $DRY "${COMMON_EXCLUDES[@]}" \
      "$REPO_ROOT/data/datasets/"        "$NAS_ROOT/data/datasets/"
    rsync -a --human-readable $DRY "${COMMON_EXCLUDES[@]}" \
      "$REPO_ROOT/data/reference/"       "$NAS_ROOT/data/reference/"
    rsync -a --human-readable $DRY "${COMMON_EXCLUDES[@]}" \
      "$REPO_ROOT/data/raw_generations/" "$NAS_ROOT/data/raw_generations/"
    [[ -f "$REPO_ROOT/INDEX.parquet" ]] && rsync -a $DRY "$REPO_ROOT/INDEX.parquet" "$NAS_ROOT/INDEX.parquet"
    audit_log "nightly" "$NAS_ROOT" 0 0
    ;;

  *)
    grep -E '^#( +|$)' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
