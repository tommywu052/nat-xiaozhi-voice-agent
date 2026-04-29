#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Start Nemotron-3-Nano-Omni vLLM server (OpenAI-compatible API)
#
# Usage:
#   ./scripts/start-nano-omni.sh              # Default: NVFP4 on port 8880
#   ./scripts/start-nano-omni.sh --bf16       # BF16 precision (needs ≥64GB VRAM)
#   ./scripts/start-nano-omni.sh --fp8        # FP8 precision (needs ≥32GB VRAM)
#
# Prerequisites:
#   - vLLM ≥ 0.20.0 with audio support: pip install vllm[audio]==0.20.0
#   - CUDA 13.0+ driver (580+)
#   - For NVFP4 (default): Blackwell GPU (GB10 / RTX PRO 6000)
#
# Reference:
#   https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano-Omni/vllm_cookbook.ipynb
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PORT="${NANO_OMNI_PORT:-8880}"
HOST="${NANO_OMNI_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${NANO_OMNI_MAX_MODEL_LEN:-131072}"

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

echo "════════════════════════════════════════════════════════════════"
echo "  Nemotron-3-Nano-Omni vLLM Server"
echo "  Model:     $MODEL"
echo "  Precision: $PRECISION"
echo "  Endpoint:  http://${HOST}:${PORT}/v1"
echo "  Served as: nemotron"
echo "════════════════════════════════════════════════════════════════"

exec vllm serve "$MODEL" \
    --served-model-name nemotron \
    --trust-remote-code \
    --dtype auto \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --media-io-kwargs '{"video":{"num_frames":512,"fps":1}}' \
    --video-pruning-rate 0.5 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 \
    $EXTRA_ARGS
