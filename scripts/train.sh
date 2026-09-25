#!/usr/bin/env bash
# Stage 4: one training condition from its recipe, through to a servable checkpoint.
#   bash scripts/train.sh configs/train/ours.yaml <base checkpoint> ['output/runs/*_collect']
# Uses the training environment (requirements-train.txt). The servable checkpoint lands in
# output/models/<recipe name>/served; serve it with scripts/serve_student.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
RECIPE="${1:?recipe yaml}"; BASE="${2:?base checkpoint}"; RUNS="${3:-output/runs/*_collect}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m uva.train.run_recipe "$RECIPE" --base "$BASE" --solved-runs $RUNS
