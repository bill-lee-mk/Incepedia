#!/bin/bash
# scripts/eval_rolling_remote.sh — rolling eval on a remote host (e.g. 225)
# while training is still producing checkpoints on the canonical NAS host
# (165 / 118 → us-south-2 NAS).
#
# Loop, every POLL_INTERVAL seconds:
#   1. ssh to canonical host, list valid ckpts under <CKPT_NAS>
#   2. for each ckpt not yet evaluated locally:
#        a. rsync ckpt (165 NAS → local on this host)
#        b. run eval_all_ckpts.py on local copy (parallel = local GPU count)
#        c. rsync resulting metrics.json back to 165 NAS
#   3. exit when all ckpts up to TARGET_TOTAL_CKPTS are evaluated
#
# Idempotent: every iteration re-uses cached metrics.json on remote NAS to
# decide what's already done. Safe to kill+relaunch (run_detached.sh wrapper).
#
# Required env or flags:
#   EXP_ID                 = exp_finephrase_repro_protC_seed42
#   REMOTE_HOST            = ubuntu@192.222.52.165 (or 118 — whoever can reach the canonical NAS)
#   REMOTE_NAS_BASE        = /lambda/nfs/us-south-2/incepedia_exp_bak/experiments/${EXP_ID}
#   REMOTE_DATA_NAS_BASE   = /lambda/nfs/us-south-2/incepedia_exp_bak/data/datasets
#   LOCAL_REPO             = /home/ubuntu/lilei/projects/Incepedia
#   POLL_INTERVAL          = 600 (10 min)
#   TARGET_TOTAL_CKPTS     = 21 (= 20 intermediate every-500 + 1 final 10013)
#   SSH_KEY                = ~/.ssh/incept_sh_rsa
#
# Usage:
#   bash scripts/run_detached.sh eval_rolling -- bash scripts/eval_rolling_remote.sh
set -euo pipefail

EXP_ID="${EXP_ID:-exp_finephrase_repro_protC_seed42}"
REMOTE_HOST="${REMOTE_HOST:-ubuntu@192.222.52.165}"
REMOTE_NAS_BASE="${REMOTE_NAS_BASE:-/lambda/nfs/us-south-2/incepedia_exp_bak/experiments/${EXP_ID}}"
REMOTE_DATA_NAS_BASE="${REMOTE_DATA_NAS_BASE:-/lambda/nfs/us-south-2/incepedia_exp_bak/data/datasets}"
LOCAL_REPO="${LOCAL_REPO:-/home/ubuntu/lilei/projects/Incepedia}"
POLL_INTERVAL="${POLL_INTERVAL:-600}"
TARGET_TOTAL_CKPTS="${TARGET_TOTAL_CKPTS:-21}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/incept_sh_rsa}"

LOCAL_EXP_DIR="${LOCAL_REPO}/experiments/${EXP_ID}"
LOCAL_CKPT_DIR="${LOCAL_EXP_DIR}/ckpt"
LOCAL_EVAL_CURVE_DIR="${LOCAL_EXP_DIR}/eval_curve"

mkdir -p "${LOCAL_CKPT_DIR}" "${LOCAL_EVAL_CURVE_DIR}"

ssh_run() {
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "${REMOTE_HOST}" "$@"
}

rsync_pull() {
    # rsync from 165's NAS path → local
    rsync -aP --delete-after \
        -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new" \
        "${REMOTE_HOST}:$1" "$2"
}

rsync_push() {
    # rsync from local → 165's NAS path
    rsync -aP \
        -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new" \
        "$1" "${REMOTE_HOST}:$2"
}

list_remote_ckpts() {
    # Echo space-sep step numbers that have a complete ckpt (model/ +
    # checkpoint_metadata.json + config.yaml) on the canonical NAS.
    ssh_run "
        ls -d '${REMOTE_NAS_BASE}/ckpt'/*/ 2>/dev/null | while read d; do
            step=\$(basename \"\$d\")
            [[ \$step =~ ^[0-9]+$ ]] || continue
            test -d \"\$d/model\" || continue
            test -f \"\$d/checkpoint_metadata.json\" || continue
            test -f \"\$d/config.yaml\" || continue
            echo \$step
        done
    " | tr '\n' ' '
}

remote_metrics_exists() {
    local step="$1"
    local step_str
    step_str=$(printf '%07d' "${step}")
    ssh_run "test -f '${REMOTE_NAS_BASE}/eval_curve/step_${step_str}/metrics.json'"
}

local_eval_ckpt() {
    local step="$1"
    local step_str
    step_str=$(printf '%07d' "${step}")

    echo ""
    echo "== STEP ${step}  $(date -u --iso-8601=seconds) =="
    echo "[rolling] pulling ckpt step=${step} from ${REMOTE_HOST}..."

    local_ckpt="${LOCAL_CKPT_DIR}/${step}/"
    mkdir -p "${local_ckpt}"
    rsync_pull "${REMOTE_NAS_BASE}/ckpt/${step}/" "${local_ckpt}"

    echo "[rolling] running eval (2-way parallel × 1 GPU)..."
    cd "${LOCAL_REPO}"
    /home/ubuntu/miniconda3/envs/incepedia/bin/python scripts/eval_all_ckpts.py \
        --config "experiments/${EXP_ID}/config.yaml" \
        --parallel-jobs 2 \
        --gpus-per-job 1 \
        --keep-hf 2>&1 | grep -vE "warnings.warn|FutureWarning|UserWarning"

    metrics="${LOCAL_EVAL_CURVE_DIR}/step_${step_str}/metrics.json"
    if [ ! -f "${metrics}" ]; then
        echo "[rolling] ERROR: eval did not produce ${metrics}; skipping push"
        return 1
    fi

    echo "[rolling] pushing metrics.json back to NAS..."
    ssh_run "mkdir -p '${REMOTE_NAS_BASE}/eval_curve/step_${step_str}'"
    rsync_push "${metrics}" "${REMOTE_NAS_BASE}/eval_curve/step_${step_str}/metrics.json"

    echo "[rolling] done step ${step}"
    # Reclaim local disk: we can drop the nanotron ckpt now (we have the metrics).
    # Keep the HF ckpt only if env says so.
    rm -rf "${local_ckpt}"
}

count_evaluated() {
    # ssh to remote, count metrics.json files
    ssh_run "ls '${REMOTE_NAS_BASE}/eval_curve'/step_*/metrics.json 2>/dev/null | wc -l" || echo 0
}

main() {
    echo "[rolling] starting rolling eval on $(hostname) for ${EXP_ID}"
    echo "[rolling] remote canonical: ${REMOTE_HOST}:${REMOTE_NAS_BASE}"
    echo "[rolling] local repo: ${LOCAL_REPO}"
    echo "[rolling] target total ckpts: ${TARGET_TOTAL_CKPTS}"
    echo "[rolling] poll interval: ${POLL_INTERVAL}s"

    while true; do
        echo ""
        echo "================================================================"
        echo "[rolling] poll @ $(date -u --iso-8601=seconds)"
        echo "================================================================"

        ckpts=($(list_remote_ckpts))
        echo "[rolling] remote has ${#ckpts[@]} ckpts: ${ckpts[*]}"

        evaluated=$(count_evaluated)
        echo "[rolling] already evaluated (metrics.json on NAS): ${evaluated}"

        if [ "${evaluated}" -ge "${TARGET_TOTAL_CKPTS}" ]; then
            echo "[rolling] all ${TARGET_TOTAL_CKPTS} ckpts evaluated. Exiting."
            break
        fi

        # Process new ckpts
        for step in "${ckpts[@]}"; do
            if remote_metrics_exists "${step}"; then
                continue
            fi
            local_eval_ckpt "${step}" || true
        done

        echo ""
        echo "[rolling] sleeping ${POLL_INTERVAL}s before next poll..."
        sleep "${POLL_INTERVAL}"
    done
}

main "$@"
