#!/bin/bash
# scripts/launch_if_gpu_free.sh — GPU-availability gate + cross-host coordination
# wrapper for resuming training. Used as the `resume_cmd` in watchdog jobs.json
# so that:
#
#   1. If LOCAL GPUs are taken (by another team's job), DO NOT crash.  Instead
#      sleep, poll, and only launch training when all required GPUs are free.
#      This is intentional: the previous disaster (Sun 15:00 China) was caused
#      by watchdog blindly relaunching nanotron while GPUs were occupied →
#      every relaunch failed instantly → max-restarts hit in 10 min → gave up,
#      another team grabbed our GPUs within minutes.
#
#   2. If ANOTHER NODE is already actively training (heartbeat in NAS lock file
#      <300s old), DO NOT also launch.  Prevents 118+165 race where both nodes
#      take step 1000 ckpt and write conflicting step 1500 ckpts back to NAS.
#
#   3. While waiting, write progress to stdout every 60s so the watchdog's
#      Mode 2 (stale log) check sees us as healthy.
#
# Configurable via env (defaults shown):
#   N_REQUIRED=8                         GPUs needed (all-or-nothing)
#   FREE_THRESHOLD_MB=2000               GPU "free" if mem.used < this MB
#   POLL_SECONDS=120                     wait between polls
#   NAS_LOCK=<NAS>/<exp>/.trainer_lock   cross-node coordination file
#   LOCK_STALE_SECONDS=300               other-host lock considered abandoned after this
#   PYTHON_BIN=python                    set to a conda-env path if conda not activated
#
# Usage in jobs.json:
#   "resume_cmd": ["bash", "scripts/launch_if_gpu_free.sh"]
#
# Set EXP_ID env var if running on a different experiment.
set -uo pipefail

N_REQUIRED="${N_REQUIRED:-8}"
FREE_THRESHOLD_MB="${FREE_THRESHOLD_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
EXP_ID="${EXP_ID:-exp_finephrase_repro_protC_seed42}"
NAS_BASE="${NAS_BASE:-/lambda/nfs/us-south-2/incepedia_exp_bak/experiments/${EXP_ID}}"
NAS_LOCK="${NAS_LOCK:-${NAS_BASE}/.trainer_lock}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-300}"

HOST="$(hostname)"
SELF_PID="$$"

log() {
    printf '[%s][gate %s pid=%d] %s\n' "$(date -u --iso-8601=seconds)" "$HOST" "$SELF_PID" "$*"
}

count_busy_gpus() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | head -n "$N_REQUIRED" \
        | awk -v t="$FREE_THRESHOLD_MB" '$1+0 > t' \
        | wc -l
}

remote_lock_status() {
    # Echos one of: STALE, MINE, OTHER:<host>:<age_s>, NONE
    if [ ! -f "$NAS_LOCK" ]; then
        echo "NONE"; return
    fi
    local mtime
    mtime="$(stat -c %Y "$NAS_LOCK" 2>/dev/null || echo 0)"
    local age=$(( $(date +%s) - mtime ))
    local host_in_lock
    host_in_lock="$(awk '{print $1; exit}' "$NAS_LOCK" 2>/dev/null)"
    if [ "$age" -gt "$LOCK_STALE_SECONDS" ]; then
        echo "STALE:${host_in_lock}:${age}"
    elif [ "$host_in_lock" = "$HOST" ]; then
        echo "MINE:${age}"
    else
        echo "OTHER:${host_in_lock}:${age}"
    fi
}

mkdir -p "$NAS_BASE" 2>/dev/null || true
log "starting; N_REQUIRED=$N_REQUIRED FREE_THRESHOLD_MB=$FREE_THRESHOLD_MB lock=$NAS_LOCK"

# ── WAIT LOOP ────────────────────────────────────────────────────────────
poll_count=0
while true; do
    poll_count=$((poll_count+1))
    busy=$(count_busy_gpus)
    lock_status=$(remote_lock_status)

    case "$lock_status" in
        OTHER:*)
            other_host="$(echo "$lock_status" | awk -F: '{print $2}')"
            age_s="$(echo "$lock_status" | awk -F: '{print $3}')"
            log "poll #${poll_count}: another node already training (${other_host}, last beat ${age_s}s ago); sleeping ${POLL_SECONDS}s"
            sleep "$POLL_SECONDS"
            continue
            ;;
        MINE:*)
            log "WARN: NAS lock already held by us — stale leftover from previous run; reclaiming"
            ;;
        STALE:*)
            stale_host="$(echo "$lock_status" | awk -F: '{print $2}')"
            stale_age="$(echo "$lock_status" | awk -F: '{print $3}')"
            log "poll #${poll_count}: NAS lock stale (held by ${stale_host}, ${stale_age}s old > ${LOCK_STALE_SECONDS}s); will reclaim"
            ;;
        NONE)
            log "poll #${poll_count}: no NAS lock present"
            ;;
    esac

    if [ "$busy" -gt 0 ]; then
        log "poll #${poll_count}: ${busy}/${N_REQUIRED} local GPUs busy (>${FREE_THRESHOLD_MB}MB used); sleeping ${POLL_SECONDS}s"
        sleep "$POLL_SECONDS"
        continue
    fi

    # All clear: take the lock + break out
    echo "${HOST} ${SELF_PID} $(date -u --iso-8601=seconds)" > "$NAS_LOCK"
    log "all ${N_REQUIRED} GPUs free + NAS lock acquired; launching training"
    break
done

# ── HEARTBEAT SIDECAR ────────────────────────────────────────────────────
# Refresh NAS lock mtime every 60s so peers know we're still training.
(
    while sleep 60; do
        if [ ! -f "$NAS_LOCK" ]; then break; fi
        echo "${HOST} ${SELF_PID} $(date -u --iso-8601=seconds)" > "$NAS_LOCK"
    done
) &
SIDECAR_PID=$!

cleanup() {
    rc=$?
    log "cleanup: killing sidecar $SIDECAR_PID + releasing NAS lock"
    kill "$SIDECAR_PID" 2>/dev/null || true
    rm -f "$NAS_LOCK" 2>/dev/null || true
    exit $rc
}
trap cleanup EXIT INT TERM

# ── LAUNCH TRAINING ──────────────────────────────────────────────────────
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate incepedia
log "exec: python scripts/run_experiment.py --config experiments/${EXP_ID}/config.yaml"

python scripts/run_experiment.py --config "experiments/${EXP_ID}/config.yaml"
TRAIN_RC=$?
log "training exited with code $TRAIN_RC"
exit $TRAIN_RC
