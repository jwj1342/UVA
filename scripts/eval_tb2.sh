#!/usr/bin/env bash
# Evaluate one policy on Terminal-Bench 2.0 (all 89 tasks; the runner verifies each task itself).
#   bash scripts/eval_tb2.sh <solo|natural|privacy_prompt|host_sanitizer> <run tag> [slice] [workers]
# Prerequisites: uva/eval/terminalbench/fetch_tb2_tasks.sh and build_tb2_images.sh.
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/env.sh
ARM="${1:?arm}"; TAG="${2:?run tag}"; SLICE="${3:-0:89}"; WORKERS="${4:-4}"
CONFIG="configs/eval/tb2_${ARM}.yaml"
ack_public_benchmark "$CONFIG"
python -m uva.eval.terminalbench.run_tb2 --tasks-dir "${TB2_TASKS_DIR:-data/terminalbench2}" \
    --sif-dir "${TB2_SIF_DIR:-containers/tb2}" --config "$CONFIG" \
    --output "output/runs/${TAG}" --slice "$SLICE" --workers "$WORKERS"
# disclosure against each task's environment-definition inventory
mkdir -p output/reports
python -m uva.eval.privacy_table --tb-tasks-dir "${TB2_TASKS_DIR:-data/terminalbench2}" \
    --arm "${ARM}=output/runs/${TAG}" --out "output/reports/privacy_${TAG}.json"
