#!/usr/bin/env bash
# Solve statistics and disclosure per policy from scored runs: solve statistics and disclosure per policy.
#   bash scripts/report.sh <pool> NAME=run[,run...] [NAME=run[,run...] ...]
# e.g. bash scripts/report.sh data/pools/verified_500.jsonl \
#        solo=output/runs/verified_solo_k0,output/runs/verified_solo_k1,output/runs/verified_solo_k2 \
#        natural=output/runs/verified_natural_k0,output/runs/verified_natural_k1,output/runs/verified_natural_k2
# Each comma-separated run is one sample of the policy over the pool; the disclosure table pools the
# consultations of all samples, and Ask% is taken over pool size x samples.
set -euo pipefail
cd "$(dirname "$0")/.."
POOL="${1:?pool}"; shift
mkdir -p output/reports
ARMS=(); for a in "$@"; do ARMS+=(--arm "$a"); done
python -m uva.eval.paper_stats passk --pool "$POOL" "${ARMS[@]}" --k 1 3
first="${1#*=}"; samples=$(( $(tr -cd , <<<"$first" | wc -c) + 1 ))
python -m uva.eval.privacy_table --pool "$POOL" --samples "$samples" "${ARMS[@]}" --out output/reports/privacy.json
