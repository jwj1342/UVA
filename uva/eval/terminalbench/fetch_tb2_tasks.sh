#!/usr/bin/env bash
# Fetch the Terminal-Bench 2.0 task metadata (instruction.md, task.toml, tests/, environment/),
# not the images. Shallow clone of the frozen 2.0 task set.
set -euo pipefail
DEST="${1:-data/terminalbench2}"
if [ -d "$DEST/.git" ]; then
    git -C "$DEST" pull --ff-only
else
    git clone --depth 1 https://github.com/harbor-framework/terminal-bench-2 "$DEST"
fi
echo "tasks: $(find "$DEST" -maxdepth 2 -name task.toml | wc -l) at $DEST"
