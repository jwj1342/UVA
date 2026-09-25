#!/usr/bin/env bash
# Pull the 89 prebuilt Terminal-Bench 2.0 task images as Apptainer SIFs.
#   bash uva/eval/terminalbench/build_tb2_images.sh [containers/tb2]
# Image list: tb2_images.tsv (task, docker_image, ...) generated from the task.toml files.
# Skips images that already exist, so a re-run resumes.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSV="${HERE}/tb2_images.tsv"
OUT_DIR="${1:-containers/tb2}"
mkdir -p "$OUT_DIR"
BUILT=0 SKIPPED=0 FAILED=0
while IFS=$'\t' read -r TASK IMAGE _REST; do
    [ "$TASK" = "task" ] && continue
    [ -z "$IMAGE" ] && continue
    SIF="$OUT_DIR/${TASK}.sif"
    if [ -f "$SIF" ]; then SKIPPED=$((SKIPPED + 1)); continue; fi
    echo "[pull] $TASK <- docker://$IMAGE"
    if apptainer build "${SIF}.tmp" "docker://$IMAGE" >/dev/null; then
        mv "${SIF}.tmp" "$SIF"; BUILT=$((BUILT + 1))
    else
        rm -f "${SIF}.tmp"; echo "[FAIL] $TASK"; FAILED=$((FAILED + 1))
    fi
done < "$TSV"
echo "built $BUILT skipped $SKIPPED failed $FAILED -> $OUT_DIR"
[ "$FAILED" -eq 0 ]
