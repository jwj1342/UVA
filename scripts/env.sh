#!/usr/bin/env bash
# Shared environment for the pipeline scripts. Fill in .env (see .env.example) or export the
# variables yourself; this file sources .env when it exists.
#
#   STUDENT_API_BASE   OpenAI-compatible endpoint serving the student (scripts/serve_student.sh)
#   CLOUD_API_BASE     OpenAI-compatible endpoint of the expert E, CLOUD_API_KEY its key
#   CLOUD_EXPERT_MODEL expert model name at that endpoint (default deepseek/deepseek-v3.2)
#   TEACHER_API_BASE   endpoint of the offline teacher R, TEACHER_API_KEY its key, TEACHER_MODEL its name
#   UVA_SIF_CACHE_DIR  directory of Apptainer images (sweb.eval.x86_64.<id>.sif, swesmith.x86_64.<repo>.sif)
#   UVA_ACK_PUBLIC_BENCHMARK=1   consulting policies send questions off the machine verbatim; say so.
_UVA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${_UVA_ROOT}/.env" ]]; then
    set -a; source "${_UVA_ROOT}/.env"; set +a
fi
export STUDENT_API_BASE="${STUDENT_API_BASE:-http://127.0.0.1:8000/v1}"
export HOSTED_VLLM_API_BASE="${STUDENT_API_BASE}"
export HOSTED_VLLM_API_KEY="${HOSTED_VLLM_API_KEY:-dummy}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export CLOUD_EXPERT_MODEL="${CLOUD_EXPERT_MODEL:-deepseek/deepseek-v3.2}"
export UVA_SIF_CACHE_DIR="${UVA_SIF_CACHE_DIR:-containers}"
export MSWEA_SIF_CACHE_DIR="${UVA_SIF_CACHE_DIR}"
export MSWEA_SINGULARITY_EXECUTABLE="${MSWEA_SINGULARITY_EXECUTABLE:-apptainer}"
export MSWEA_SILENT_STARTUP=1
export TMPDIR="${TMPDIR:-/tmp}"

ack_public_benchmark() {
    # A consulting config transmits the student's question verbatim. That is acceptable on the
    # public benchmarks of the paper and on nothing else; the operator states it explicitly.
    local config="$1"
    if grep -qE '^\s*cloud_reply:\s*true' "$config" && ! grep -qE '^\s*suppress_ask:\s*true' "$config"; then
        [[ "${UVA_ACK_PUBLIC_BENCHMARK:-0}" == "1" ]] || {
            echo "[FAIL] $config sends questions off-host. Set UVA_ACK_PUBLIC_BENCHMARK=1 for a public benchmark." >&2
            exit 4; }
        [[ -n "${CLOUD_API_BASE:-}" && ( -n "${CLOUD_API_KEY:-}" || -n "${CLOUD_API_KEY_FILE:-}" ) ]] || {
            echo "[FAIL] set CLOUD_API_BASE and CLOUD_API_KEY (or CLOUD_API_KEY_FILE)" >&2; exit 4; }
    fi
}
