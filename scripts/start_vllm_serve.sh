#!/bin/bash
# Qwen3-Omni vLLM Serve 啟動腳本 (Thinker only, 純文字)
# 適用於 Qwen3 ASR 模式（搭配 CosyVoice TTS）
#
# 使用方法:
#   wsl -d Ubuntu-22.04
#   bash scripts/start_vllm_serve.sh
#
# 環境變數:
#   QWEN3_OMNI_MODEL_PATH  模型路徑 (預設: ~/qwen3-omni/Qwen3-Omni-30B-A3B-Instruct)
#   VLLM_PORT               服務端口 (預設: 8901)

export PATH=/usr/local/cuda/bin:/usr/bin:/usr/sbin:/bin:/sbin:/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

source ~/qwen3-omni/venv/bin/activate

MODEL_PATH="${QWEN3_OMNI_MODEL_PATH:-$HOME/qwen3-omni/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${VLLM_PORT:-8901}"

echo "========================================="
echo "  Qwen3-Omni vLLM Serve (Thinker only)"
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "========================================="

vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --allowed-local-media-path / \
    -tp 1
