#!/usr/bin/env bash
# Evaluate one policy on SWE-bench Verified: rollouts, then scoring.
#   bash scripts/eval_verified.sh <solo|natural|privacy_prompt|host_sanitizer> <run tag> [slice] [workers]
# Ours = `natural` with the trained checkpoint served (scripts/serve_student.sh). For pass@3 run
# three times with server seeds 20260716/17/18 and distinct run tags.
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/env.sh
ARM="${1:?arm}"; TAG="${2:?run tag}"; SLICE="${3:-0:500}"; WORKERS="${4:-3}"
POOL="${POOL:-data/pools/verified_500.jsonl}"
CONFIG="configs/eval/verified_${ARM}.yaml"
ack_public_benchmark "$CONFIG"
OUT="output/runs/${TAG}"; mkdir -p "$OUT"
python -m uva.harness.run_swebench \
    --subset "$POOL" --split test --slice "$SLICE" --config "$CONFIG" \
    --output "$OUT" --workers "$WORKERS" --environment-class singularity
python -m uva.eval.score_verified --preds "$OUT/preds.json" --output "$OUT/eval_results.json" \
    --workers "${EVAL_WORKERS:-8}"
echo "[eval] $OUT scored"
