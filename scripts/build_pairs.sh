#!/usr/bin/env bash
# Stages 2-3: candidates (teacher rewrite + pre-replay checks) -> replay gate -> master set.
#   TARGET=abstract  bash scripts/build_pairs.sh <pool.jsonl> <collection run tag>
#   TARGET=pseudonym bash scripts/build_pairs.sh ...   (Host-SFT ablation; separate pairs dir)
# Needs the student served and both the expert and the teacher configured (scripts/env.sh).
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/env.sh
POOL="${1:?pool jsonl}"; TAG="${2:?collection run tag}"
TARGET="${TARGET:-abstract}"
PAIRS_DIR="output/pairs"; [[ "$TARGET" == "pseudonym" ]] && PAIRS_DIR="output/pairs_pseudonym"
COLLECT="output/runs/${TAG}"; REPLAY="output/runs/${TAG}_replay_${TARGET}"
CAND="${PAIRS_DIR}/${TAG}_candidates.jsonl"; VERIFIED="${PAIRS_DIR}/${TAG}_verified.jsonl"
REPLAY_CONFIG=configs/replay/swesmith_replay.yaml
ack_public_benchmark "$REPLAY_CONFIG"
mkdir -p "$PAIRS_DIR"

echo "[pairs] 2/5 candidates ($TARGET)"
python -m uva.data.build_candidates --runs "$COLLECT" --out "$CAND" --target "$TARGET" \
    ${TEACHER_MODEL:+--rewriter-model "$TEACHER_MODEL"}
cat "$CAND" >> "${PAIRS_DIR}/all_candidates.jsonl"      # every pre-replay candidate (no-replay-gate ablation)
echo "[pairs] 3/5 replay every candidate (q+ arm and prefix-only control)"
python -m uva.data.replay_verify --candidates "$CAND" --config "$REPLAY_CONFIG" \
    --records "$POOL" --runs-root output/runs --out-dir "$REPLAY" \
    --audit-subset "${AUDIT_SUBSET:-10}" --audit-repeats "${AUDIT_REPEATS:-2}"
echo "[pairs] 4/5 score both arms"
for armdir in "$REPLAY"/qplus_k* "$REPLAY"/noask_k*; do   # includes the qplus_k*_rep* audit arms
    [[ -d "$armdir" && -f "$armdir/preds.json" ]] || continue
    python -m uva.eval.score_swesmith --preds "$armdir/preds.json" --records "$POOL" \
        --output "$armdir/eval_results.json" --workers "${EVAL_WORKERS:-8}"
done
python -m uva.data.replay_verify --join --candidates "$CAND" --out-dir "$REPLAY" --out "$VERIFIED"
echo "[pairs] 5/5 master set"
python -m uva.data.pairs_master append --batch "$TAG" --collect-dir "$COLLECT" \
    --candidates "$CAND" --verified "$VERIFIED" --pairs-dir "$PAIRS_DIR"
python -m uva.data.pairs_master report --pairs-dir "$PAIRS_DIR"
