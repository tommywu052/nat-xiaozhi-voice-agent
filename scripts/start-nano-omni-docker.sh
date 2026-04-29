#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Start Nemotron-3-Nano-Omni via Docker (vllm/vllm-openai container)
#
# Usage:
#   ./scripts/start-nano-omni-docker.sh             # NVFP4 (default, Blackwell)
#   ./scripts/start-nano-omni-docker.sh --bf16      # BF16 (H100 / ≥64GB)
#   ./scripts/start-nano-omni-docker.sh --fp8       # FP8
#
# Replaces the Gemma4 container (vllm-gemma4-lite) on the same port.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONTAINER_NAME="vllm-nano-omni"
PORT="${NANO_OMNI_PORT:-8880}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

PRECISION="${1:-nvfp4}"

case "$PRECISION" in
  --bf16|bf16)
    MODEL="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    EXTRA_ARGS=""
    ;;
  --fp8|fp8)
    MODEL="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
    EXTRA_ARGS=""
    ;;
  --nvfp4|nvfp4|*)
    MODEL="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    EXTRA_ARGS="--kv-cache-dtype fp8 --moe-backend cutlass --max-num-seqs 8 --gpu-memory-utilization 0.85"
    ;;
esac

echo "Stopping existing Gemma4 container if running..."
docker stop vllm-gemma4-lite 2>/dev/null || true
docker rm vllm-gemma4-lite 2>/dev/null || true

echo "Stopping existing Nano Omni container if running..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "════════════════════════════════════════════════════════════════"
echo "  Nemotron-3-Nano-Omni Docker Server"
echo "  Model:     $MODEL"
echo "  Image:     $IMAGE"
echo "  Port:      $PORT"
echo "════════════════════════════════════════════════════════════════"

docker run -d \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    -p "${PORT}:${PORT}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    --ipc=host \
    --restart unless-stopped \
    "$IMAGE" \
    --model "$MODEL" \
    --served-model-name nemotron \
    --trust-remote-code \
    --dtype auto \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --media-io-kwargs '{"video":{"num_frames":512,"fps":1}}' \
    --video-pruning-rate 0.5 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 \
    $EXTRA_ARGS

echo ""
echo "Container '$CONTAINER_NAME' started. Monitor with:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "API endpoint: http://localhost:${PORT}/v1"
echo "Model name:   nemotron"
