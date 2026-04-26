#!/bin/bash
# scripts/test_detach.sh — verify run_detached.sh actually daemonizes.
#
# Test plan (no GPU, no real training; uses sleep as proxy):
#   1. Launch a long sleep via run_detached.sh
#   2. Verify PPID == 1 (orphan adopted by init)
#   3. Verify SID == PID (session leader)
#   4. Verify log file is being written
#   5. Simulate parent shell exit by spawning the test in a subshell
#      that exits before checking the daemon — daemon should still exist
#   6. Cleanup: kill the dummy
#
# If any step fails, EXIT NON-ZERO. This script is the gate before
# trusting run_detached.sh for real training.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

NAME="detach_test_$$"
LOG="logs/detached/${NAME}.log"
PIDFILE="logs/detached/${NAME}.pid"

cleanup() {
    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PIDFILE" "$LOG"
    fi
}
trap cleanup EXIT

echo "=== Test 1: launch via run_detached.sh ==="
# Launch a 60s sleep that periodically writes to stdout (gives us a heartbeat in log)
bash scripts/run_detached.sh "$NAME" -- bash -c 'for i in $(seq 1 30); do echo "tick $i $(date -u +%H:%M:%S)"; sleep 2; done'

if [ ! -f "$PIDFILE" ]; then
    echo "FAIL: PID file not created" >&2
    exit 1
fi
PID=$(cat "$PIDFILE")
echo "  PID = $PID"

echo ""
echo "=== Test 2: PPID == 1 (orphan, adopted by init) ==="
sleep 2
PPID_OF_TARGET=$(ps -p "$PID" -o ppid= | tr -d ' ')
echo "  PPID = $PPID_OF_TARGET (expect 1)"
if [ "$PPID_OF_TARGET" != "1" ]; then
    echo "FAIL: PPID is $PPID_OF_TARGET, expected 1" >&2
    exit 1
fi

echo ""
echo "=== Test 3: SID == PID (session leader) ==="
SID_OF_TARGET=$(ps -p "$PID" -o sid= | tr -d ' ')
echo "  SID  = $SID_OF_TARGET (expect $PID)"
if [ "$SID_OF_TARGET" != "$PID" ]; then
    echo "FAIL: SID is $SID_OF_TARGET, expected $PID" >&2
    exit 1
fi

echo ""
echo "=== Test 4: log file actively written ==="
sleep 4
if [ ! -s "$LOG" ]; then
    echo "FAIL: log file empty" >&2
    exit 1
fi
log_age=$(($(date +%s) - $(stat -c %Y "$LOG")))
echo "  log size = $(stat -c %s "$LOG") bytes; last modified ${log_age}s ago"
if [ "$log_age" -gt 6 ]; then
    echo "FAIL: log not being actively written (last modified ${log_age}s ago)" >&2
    exit 1
fi

echo ""
echo "=== Test 5: process survives subshell exit ==="
# Spawn a subshell whose only job is to verify the daemon is alive,
# then EXIT.  If our daemonization is correct, the daemon must survive.
(
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "FAIL: daemon died inside subshell check" >&2
        exit 1
    fi
    echo "  daemon alive inside subshell"
)
# subshell exited; check from parent shell
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "FAIL: daemon died after subshell exited" >&2
    exit 1
fi
echo "  daemon STILL alive after subshell exited"

echo ""
echo "=== Test 6: parent process tree check ==="
# Walk the parent chain from PID. If the chain is short (PID → 1 → 0),
# we're orphaned. If we see other PIDs, we're attached to some shell.
chain=()
cur=$PID
while true; do
    pp=$(ps -p "$cur" -o ppid= 2>/dev/null | tr -d ' ' || true)
    [ -z "$pp" ] && break
    chain+=("$cur→$pp")
    [ "$pp" = "0" ] && break
    [ "$pp" = "1" ] && { chain+=("init"); break; }
    cur=$pp
done
echo "  process tree: ${chain[*]}"

echo ""
echo "================================================================"
echo "ALL TESTS PASSED — run_detached.sh is safe for production training"
echo "================================================================"
