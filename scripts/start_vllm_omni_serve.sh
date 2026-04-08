#!/bin/bash
# Qwen3-Omni vLLM-Omni Serve 啟動腳本 (含 Talker TTS)
# 單 GPU 配置，async_chunk 三階段分段並行
#
# 使用方法:
#   wsl -d Ubuntu-22.04
#   bash scripts/start_vllm_omni_serve.sh
#
# 環境變數 (可在 .env 或 shell 中設定):
#   QWEN3_OMNI_MODEL_PATH  模型路徑 (預設: ~/qwen3-omni/Qwen3-Omni-30B-A3B-Instruct)
#   STAGE_CONFIG_PATH       stage config 路徑 (預設: 同目錄下 qwen3_omni_single_gpu_async.yaml)
#   VLLM_PORT               服務端口 (預設: 8901)

export PATH=/usr/local/cuda/bin:/usr/bin:/usr/sbin:/bin:/sbin:/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

source ~/qwen3-omni/venv/bin/activate

MODEL_PATH="${QWEN3_OMNI_MODEL_PATH:-$HOME/qwen3-omni/Qwen3-Omni-30B-A3B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_CONFIG="${STAGE_CONFIG_PATH:-$SCRIPT_DIR/qwen3_omni_single_gpu_async.yaml}"
PORT="${VLLM_PORT:-8901}"

echo "========================================="
echo "  Qwen3-Omni vLLM-Omni Serve (Single GPU)"
echo "  Mode: async_chunk (3-stage pipeline streaming)"
echo "  Model: $MODEL_PATH"
echo "  Stage Config: $STAGE_CONFIG"
echo "  Port: $PORT"
echo "========================================="
echo ""
echo "async_chunk 模式: Thinker/Talker/Code2Wav 分段並行"
echo "  支援 modalities=[text,audio] + stream=true"
echo "  Audio TTFT ≈ 2 秒"
echo "========================================="

vllm serve "$MODEL_PATH" \
    --omni \
    --stage-configs-path "$STAGE_CONFIG" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --allowed-local-media-path / \
    --stage-init-timeout 600 \
    2>&1
