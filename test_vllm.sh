#!/bin/bash
#SBATCH -J vlmhub_vllm
#SBATCH -o vlmhub_vllm.o%j
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH --mem=32GB
#SBATCH --gpus-per-node=1

# Serve a models.json vllm model on a GPU node and run tests/ against it
# Usage, from the repo root:
#   sbatch test_vllm.sh qwen3vl-4b
#   sbatch --gpus-per-node=ada:2 --mem=128GB test_vllm.sh gpt-oss-120b   # 2 GPUs
#   sbatch --gpus-per-node=volta:1 test_vllm.sh qwen3vl-8b               # V100: no FP8/MXFP4 models
# Optional .env: VENV_PATH, HF_HOME, XDG_CACHE_HOME. GPU types: sinfo -o "%15N %10T %20G"

set -euo pipefail
MODEL_NAME="${1:-qwen3vl-4b}"
LOG="vllm_server.o${SLURM_JOB_ID}"

# ── Setup ─────────────────────────────────────────────────────────────────────
if [[ -f .env ]]; then set -a; source .env; set +a; fi
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi
if [[ -n "${XDG_CACHE_HOME:-}" ]]; then export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"; fi

# ── Model and port ────────────────────────────────────────────────────────────
# HF repo id, e.g. qwen3vl-4b -> Qwen/Qwen3-VL-4B-Instruct
MODEL_ID=$(python -c "from vlmhub.utils.config import Config; print(Config().get_client_by_name('vllm/$MODEL_NAME')['model_id'])")

# Free port, passed to the tests as a registry override (lets jobs share a node)
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
OVERRIDE=$(mktemp --suffix=.json)
echo "{\"hostings\": {\"vllm\": {\"api_base\": \"http://127.0.0.1:$PORT/v1\"}}}" > "$OVERRIDE"

# ── Server ────────────────────────────────────────────────────────────────────
# All GPUs of the job; context capped at 32K (default: the model's full context, which may not fit)
# Qwen3.x: add --reasoning-parser qwen3 to drop the thinking from answers
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
vllm serve "$MODEL_ID" --port "$PORT" --max-model-len 32768 --tensor-parallel-size "$NUM_GPUS" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID' EXIT

# Wait until ready (the first run also downloads the weights)
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    kill -0 $SERVER_PID || { echo "vllm server failed, see $LOG"; exit 1; }
    sleep 10
done

# ── Tests ─────────────────────────────────────────────────────────────────────
# Comment out what you don't need; a failing test doesn't stop the rest
run() { python "tests/$1.py" "${@:2}" --models-path "$OVERRIDE" || echo "$1.py failed"; }
CLIENT="vllm/$MODEL_NAME"

# Tasks, image models: COCO captioning and VQAv2, printed next to the references
# run test_task_captioning --client "$CLIENT"
# run test_task_vqa        --client "$CLIENT"

# Smoke test, image models: one image, one prompt
# run test_clients         --clients "$CLIENT"

# Judge, any model: grades 4 answers on a 1-5 rubric
run test_judge           --client "$CLIENT"
