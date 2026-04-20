#!/usr/bin/env bash
# Run lighteval on a model via the Incepedia EvalRunner.
#
# Usage:
#   scripts/run_eval.sh <model> <output_dir> [task_group] [max_samples]
#
# Examples:
#   scripts/run_eval.sh HuggingFaceTB/cosmo-1b experiments/_sanity/eval
#   scripts/run_eval.sh HuggingFaceTB/cosmo-1b experiments/_sanity/eval early-signal 500
#
# Env overrides:
#   NUM_PROCESSES=8   BATCH_SIZE=16   MAIN_PORT=29600

set -euo pipefail

MODEL="${1:?usage: scripts/run_eval.sh <model> <output_dir> [task_group] [max_samples]}"
OUTPUT_DIR="${2:?output_dir required}"
TASK_GROUP="${3:-early-signal}"
MAX_SAMPLES="${4:-1000}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAIN_PORT="${MAIN_PORT:-29600}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METRICS_PATH="${OUTPUT_DIR%/eval}/metrics.json"

python - <<PY
from incepedia.eval.runner import EvalRunner

runner = EvalRunner(
    model="${MODEL}",
    output_dir="${OUTPUT_DIR}",
    task_group="${TASK_GROUP}",
    num_processes=${NUM_PROCESSES},
    batch_size=${BATCH_SIZE},
    max_samples=${MAX_SAMPLES} if "${MAX_SAMPLES}".isdigit() else None,
    main_process_port=${MAIN_PORT},
)
scores = runner.run()
metrics_path = runner.write_metrics_json("${METRICS_PATH}")
print("[eval] wrote metrics to:", metrics_path)
for task, score in sorted(scores.items()):
    print(f"  {task:24s}  {score:.4f}")
PY
