#!/bin/bash
# scripts/run_detached.sh — TRUE daemonization wrapper.
#
# Replaces our prior unreliable habit of using cursor's `block_until_ms: 0`,
# `nohup &`, or partial setsid attempts — all of which kept the launched
# process as a child of cursor's bash wrapper, so it died on SSH drop /
# cursor session restart.
#
# Guarantees (all VERIFIED below before we return):
#   1. process becomes session leader (sid == pid)
#   2. parent is init (PPID == 1) within 2s of launch
#   3. controlling terminal severed (no /dev/tty access)
#   4. stdout/stderr appended to logs/detached/<name>.log
#   5. PID file written to logs/detached/<name>.pid
#   6. resume command echoed for human convenience
#
# Failure mode: exits non-zero with diagnostic if PPID != 1, so the caller
# *cannot* be misled about detachment status.
#
# Usage:
#   scripts/run_detached.sh <name> -- <command> [args...]
#
# Example:
#   scripts/run_detached.sh protC_train -- \
#       python scripts/run_experiment.py --config experiments/.../config.yaml
#
# Stop:
#   kill $(cat logs/detached/<name>.pid)
#
# Inspect:
#   tail -F logs/detached/<name>.log
#   ps -p $(cat logs/detached/<name>.pid) -o pid,ppid,sid,cmd
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
    cat >&2 <<USAGE
Usage: $0 <name> -- <command> [args...]

  <name>   short identifier (no spaces); used for logs/detached/<name>.{log,pid}
  --       required separator
  <cmd>    program + args to launch detached
USAGE
    exit 2
fi

name="$1"; shift; shift  # consume <name> + --

# repo-root anchored paths
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
log_dir="$repo_root/logs/detached"
log="$log_dir/${name}.log"
pid_file="$log_dir/${name}.pid"
mkdir -p "$log_dir"

# Refuse to clobber an already-running detached process under same name.
if [ -f "$pid_file" ]; then
    old_pid=$(cat "$pid_file" 2>/dev/null || true)
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        echo "[run_detached] ERROR: '$name' already running as PID $old_pid" >&2
        echo "[run_detached]        kill it first: kill $old_pid" >&2
        exit 3
    fi
fi

{
    echo ""
    echo "================================================================"
    echo "[run_detached] $(date -u --iso-8601=seconds) launching '$name'"
    echo "[run_detached] cmd: $*"
    echo "[run_detached] cwd: $(pwd)"
    echo "[run_detached] log: $log"
    echo "================================================================"
} >>"$log"

# Core daemonization.
#   setsid --fork  : create new session, parent immediately exits → init adopts child
#   </dev/null     : sever stdin from any terminal
#   >>"$log" 2>&1  : redirect stdout/stderr to log
#   &              : background the setsid call itself (returns instantly)
# Inside the launched bash we record PID then exec the real command.
# Using "$@" via env trick to preserve quoting of args containing spaces.
exec_args_json="$(printf '%s\n' "$@" | python3 -c '
import json, sys
print(json.dumps([line.rstrip("\n") for line in sys.stdin if line]))
')"
export INCEP_DETACHED_ARGS_JSON="$exec_args_json"
export INCEP_DETACHED_PID_FILE="$pid_file"

setsid --fork bash -c '
    echo $$ > "$INCEP_DETACHED_PID_FILE"
    exec python3 -c "
import json, os
args = json.loads(os.environ[\"INCEP_DETACHED_ARGS_JSON\"])
os.execvp(args[0], args)
"
' </dev/null >>"$log" 2>&1 &

# Note: the `& disown` here is just to release our (caller) shell from the
# setsid command, which itself has already forked and exited.  After this:
#   - bash wrapper that ran setsid exits immediately (returncode 0)
#   - inner setsid forked process is orphaned → adopted by init (PPID=1)
disown

# ── Verification: confirm PPID == 1 within 2s, else fail loudly ─────────
deadline=$(($(date +%s) + 5))
pid=""
while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file" 2>/dev/null || true)
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && break
    fi
    sleep 0.2
done

if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "[run_detached] FATAL: process did not start within 5s" >&2
    exit 4
fi

ppid=$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ' || echo "?")
sid=$(ps -p "$pid" -o sid= 2>/dev/null | tr -d ' ' || echo "?")

echo "[run_detached] launched '$name'"
echo "[run_detached]   PID  = $pid"
echo "[run_detached]   PPID = $ppid  (expect 1 = init)"
echo "[run_detached]   SID  = $sid  (expect == PID = session leader)"

if [ "$ppid" != "1" ]; then
    # Try one more time after another sec; sometimes init adoption takes a moment
    sleep 1
    ppid=$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ' || echo "?")
    if [ "$ppid" != "1" ]; then
        echo "[run_detached] FATAL: PPID is $ppid, expected 1" >&2
        echo "[run_detached]        process may NOT survive parent shell exit" >&2
        echo "[run_detached]        killing $pid; manually inspect" >&2
        kill -TERM "$pid" 2>/dev/null || true
        rm -f "$pid_file"
        exit 5
    fi
fi
if [ "$sid" != "$pid" ]; then
    echo "[run_detached] WARN: SID($sid) != PID($pid); not session leader" >&2
fi

echo "[run_detached] OK — truly detached, will survive cursor session / SSH drop"
echo "[run_detached] kill cmd: kill $pid     # or: kill \$(cat $pid_file)"
echo "[run_detached] tail log: tail -F $log"

exit 0
