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
#   NUM_PROCESSES=8   MAIN_PORT=29600   MIXED_PRECISION=bf16

set -euo pipefail

MODEL="${1:?usage: scripts/run_eval.sh <model> <output_dir> [task_group] [max_samples]}"
OUTPUT_DIR="${2:?output_dir required}"
TASK_GROUP="${3:-early-signal}"
MAX_SAMPLES="${4:-500}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PORT="${MAIN_PORT:-29600}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# Infer where to write metrics.json: the parent of the eval/ dir, or next to it.
if [[ "$OUTPUT_DIR" == */eval ]]; then
  METRICS_PATH="${OUTPUT_DIR%/eval}/metrics.json"
else
  METRICS_PATH="${OUTPUT_DIR}/metrics.json"
fi

python - <<PY
from incepedia.eval.runner import EvalRunner

runner = EvalRunner(
    model="${MODEL}",
    output_dir="${OUTPUT_DIR}",
    task_group="${TASK_GROUP}",
    num_processes=${NUM_PROCESSES},
    main_process_port=${MAIN_PORT},
    mixed_precision="${MIXED_PRECISION}",
    max_samples=${MAX_SAMPLES} if "${MAX_SAMPLES}".strip().lower() not in ("none","") else None,
)
scores = runner.run()
metrics_path = runner.write_metrics_json("${METRICS_PATH}")
print("[eval] wrote metrics to:", metrics_path)
for task, score in sorted(scores.items()):
    print(f"  {task:24s}  {score:.4f}")
PY
