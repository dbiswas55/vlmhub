#!/bin/bash
#SBATCH -J vlmhub_vllm
#SBATCH -o vlmhub_vllm.o%j
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gpus-per-node=1

# Serves a vllm model from models.json on a GPU node, then runs test_captioning and test_vqa from tests/ against it.
#
# Usage, from the repo root:
#   sbatch test_vllm.sh qwen3vl-4b
#   sbatch --gpus-per-node=ada:1 --mail-user=you@example.com --mail-type=FAIL,END test_vllm.sh qwen3vl-4b
#
# Optional .env settings: VENV_PATH (env with vlmhub + vllm), HF_HOME (model weights),
# XDG_CACHE_HOME (vllm/triton compile caches). Unset, they use the submitting shell's env and ~/.cache.
# List GPU types with: sinfo -o "%15N %10T %20G". On GPUs below sm_80 (e.g. V100), add --dtype half to `vllm serve`.

set -euo pipefail

MODEL_NAME="${1:-qwen3vl-4b}"

cd "$SLURM_SUBMIT_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi
if [[ -n "${XDG_CACHE_HOME:-}" ]]; then export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"; fi

# HF repo id of the model, e.g. qwen3vl-4b -> Qwen/Qwen3-VL-4B-Instruct
MODEL_ID=$(python -c "from vlmhub.utils.config import Config; print(Config().get_client_by_name('vllm/$MODEL_NAME')['model_id'])")

# A free port, so two jobs on one node don't collide; the test finds it through a registry override
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
OVERRIDE=$(mktemp --suffix=.json)
echo "{\"hostings\": {\"vllm\": {\"api_base\": \"http://127.0.0.1:$PORT/v1\"}}}" > "$OVERRIDE"

# Start the server in the background and stop it when the job ends
vllm serve "$MODEL_ID" --port "$PORT" --max-model-len 16384 > "vllm_server.o${SLURM_JOB_ID}" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID' EXIT

# Wait until the server is ready (the first run also downloads the weights)
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    kill -0 $SERVER_PID || { echo "vllm server failed, see vllm_server.o${SLURM_JOB_ID}"; exit 1; }
    sleep 10
done

# The tests share the server; a failing test doesn't stop the ones after it
CLIENT="vllm/$MODEL_NAME"
python tests/test_captioning.py --client  "$CLIENT" --models-path "$OVERRIDE" || echo "test_captioning.py failed"
python tests/test_vqa.py        --client  "$CLIENT" --models-path "$OVERRIDE" || echo "test_vqa.py failed"

# Quick smoke test (one image, one prompt); uncomment to run it too
# python tests/test_clients.py  --clients "$CLIENT" --models-path "$OVERRIDE" || echo "test_clients.py failed"
