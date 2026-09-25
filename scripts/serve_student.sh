#!/usr/bin/env bash
# Serve the student with vLLM under the model name the configs request (hosted_vllm/Qwen3.5-9B).
# A trained checkpoint is served under the SAME name, so the eval configs need no change.
#   bash scripts/serve_student.sh <checkpoint dir> [port] [max-model-len] [seed]
# The seed is the sampling seed of the server; the three Verified samples of a policy use
# 20260716, 20260717 and 20260718.
set -euo pipefail
MODEL_DIR="${1:?checkpoint dir}"; PORT="${2:-8000}"; MAX_LEN="${3:-131072}"; SEED="${4:-20260716}"
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name Qwen3.5-9B --port "$PORT" \
    --tensor-parallel-size 1 --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.92 \
    --trust-remote-code --dtype auto --seed "$SEED"
