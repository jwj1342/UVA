#!/usr/bin/env bash
# Stage 1: collect probed consultations on a slice of the SWE-smith training pool, then score.
#   bash scripts/collect.sh <pool.jsonl> <slice a:b> <run tag> [workers]
# Needs the student served (STUDENT_API_BASE) and the expert configured (scripts/env.sh).
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/env.sh
POOL="${1:?pool jsonl}"; SLICE="${2:?slice a:b}"; TAG="${3:?run tag}"; WORKERS="${4:-3}"
CONFIG=configs/collect/swesmith_probe.yaml
ack_public_benchmark "$CONFIG"
OUT="output/runs/${TAG}"
mkdir -p "$OUT"
python -m uva.harness.run_swebench \
    --subset "$POOL" --split train --slice "$SLICE" --config "$CONFIG" \
    --output "$OUT" --workers "$WORKERS" --environment-class singularity
python -m uva.eval.score_swesmith --preds "$OUT/preds.json" --records "$POOL" \
    --output "$OUT/eval_results.json" --workers "${EVAL_WORKERS:-8}"
echo "[collect] $OUT scored"
