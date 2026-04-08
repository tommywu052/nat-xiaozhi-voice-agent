# Xiaozhi Voice Agent — Qwen3-Omni E2E Edition (NAT Plugin)

基於 NVIDIA NeMo Agent Toolkit (NAT) 的即時語音對話 Agent，相容 [小智 ESP32](https://github.com/78/xiaozhi-esp32) / [py-xiaozhi](https://github.com/zhayujie/py-xiaozhi) 客戶端協議。

採用 **Qwen3-Omni 端到端語音模式**——單次 API 呼叫同時完成語音理解、文字推理、語音合成，將 Thinker 呼叫從 3 次降至 1 次，不需要外部 ASR 或 TTS 服務。

## 核心特性

- **端到端語音對話** — Audio In → 單次 Qwen3-Omni API → Audio Out，無需 ASR / TTS 外部服務
- **async_chunk Streaming** — Thinker / Talker / Code2Wav 三階段分段並行，~2s 首音延遲
- **Native Tool Calling** — Qwen3-Omni 原生 `<tool_call>` 標籤，非工具對話零額外延遲
- **NAT 插件架構** — 0 行 NAT 原始碼修改，`pip install` 即生效
- **100% 客戶端相容** — ESP32 / py-xiaozhi 零修改直接遷移

## E2E Streaming (async_chunk)

vLLM-Omni 的 `async_chunk` 模式讓三個階段分段並行：

```
User Audio ──► [Thinker: 理解 + 推理] ──text tokens──►
                  [Talker: 語音 tokens] ──speech tokens──►
                     [Code2Wav: PCM 音訊] ──audio chunks──► Client
```

文字 token 約在 ~1.3s 開始串流，音訊約在 ~2s 到達客戶端，不需要等待完整回覆生成。

## E2E Native Tool Calling

在 E2E streaming 中，Qwen3-Omni 可以原生輸出 `<tool_call>` 標籤呼叫工具：

```
語音輸入 → Thinker 開始生成 text stream
  ├─ 偵測到 <tool_call> → 中斷音訊 → 執行工具
  │                      → 帶結果重新呼叫 E2E → 輸出語音回答
  └─ 未偵測到 → 正常串流音訊 (零額外延遲)
```

非工具對話完全不受影響，只有需要工具時才會多一次 E2E 呼叫。

## NAT 四大擴展點運用

| 擴展機制 | 用途 | 達成效果 |
|---------|------|---------|
| `@register_front_end` | 自定義 WebSocket 伺服器，替換 NAT 預設 FastAPI | 客戶端 100% 相容原小智協定 |
| `@register_function` | LangGraph StateGraph Agent 取代預設 tool_calling_agent | Agent 和語音管線同進程直連，零代理延遲 |
| `entry_points` | pyproject.toml 宣告 nat.components + nat.front_ends | `pip install` 即生效，NAT 升級不覆蓋自定義碼 |
| `YAML _type` | 統一 YAML 配置管理 Front End + Pipeline + Agent + Tools | 一個檔案管理完整語音管線 |

## 成果

| 指標 | 數值 |
|------|------|
| Python 模組 | 18 個 (含 E2E pipeline + tool executor) |
| NAT 擴展點 | 4 個 (front_end + function + entry_points + YAML _type) |
| 修改 NAT 原始碼 | **0 行** |
| 外部依賴 | **僅 vLLM-Omni**（無需 ASR 模型 / TTS 服務） |
| 客戶端修改 | **零修改**——ESP32 / py-xiaozhi 直接遷移 |

## 架構

```
客戶端 (ESP32 / py-xiaozhi)
    │  WebSocket (Opus audio + JSON control)
    ▼
┌──────────────────────────────────────────────┐
│  Xiaozhi Voice Agent (NAT Plugin)            │
│                                              │
│  ┌─────────┐                                 │
│  │  Silero  │  語音活動偵測                    │
│  │   VAD    │                                │
│  └────┬─────┘                                │
│       │ PCM → WAV → file:// URL              │
│       ▼                                      │
│  ┌──────────────────────────────────────┐    │
│  │  vLLM-Omni (Qwen3-Omni 30B-A3B)    │    │
│  │                                      │    │
│  │  Stage 0: Thinker (理解 + 推理)      │    │
│  │     ├─ text stream ──► <tool_call>?  │    │
│  │     │    ├─ Yes → Tool Executor      │    │
│  │     │    └─ No  → 直接輸出           │    │
│  │     ▼                                │    │
│  │  Stage 1: Talker (語音 token)        │    │
│  │     ▼                                │    │
│  │  Stage 2: Code2Wav (PCM 音訊)       │    │
│  └──────────────────────────────────────┘    │
│       │                                      │
│       ▼  audio chunks → resample → Opus      │
│  ┌──────────┐                                │
│  │ E2E Tool │  current_datetime / get_weather│
│  │ Executor │  get_lunar / wiki_search       │
│  └──────────┘                                │
└──────────────────────────────────────────────┘
```

## 前置需求

### 系統

- Windows 11 + WSL2 Ubuntu 22.04 (已在 DGX Spark / RTX PRO 6000 Blackwell 驗證)
- Python 3.11 / 3.12 / 3.13
- NVIDIA GPU + CUDA 12.8

### 硬體需求

| 項目 | 規格 |
|------|------|
| GPU VRAM | ~79–98 GB (Qwen3-Omni 30B-A3B BF16，async_chunk 三階段共用 GPU) |
| 推薦 GPU | NVIDIA RTX PRO 6000 Blackwell (~98GB) / DGX Spark |
| CUDA | 12.8 |
| vLLM | 0.19.0+ (含 vLLM-Omni) |

### Qwen3-Omni 模型

```
Qwen3-Omni-30B-A3B-Instruct
├── Thinker (理解 + 推理)
│   ├── ASR: 音訊輸入 → 文字 (支援 19 種語言)
│   ├── Vision: 圖片/影片理解
│   └── LLM: 文字推理 + 對話
└── Talker (語音輸出)
    └── TTS: 文字 → 語音 (支援 10 種語言, 3 種聲音)
```

| 聲音 | 性別 | 描述 |
|------|------|------|
| Chelsie | 女 | 甜蜜、柔和、溫暖清澈 |
| Ethan | 男 | 明亮、充滿活力、溫暖親切 |
| Aiden | 男 | 溫暖、隨和的美式口音 |

### 模型下載

```bash
wsl -d Ubuntu-22.04

mkdir -p ~/qwen3-omni && cd ~/qwen3-omni

# 下載 Qwen3-Omni 模型 (~60GB)
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct', local_dir='Qwen3-Omni-30B-A3B-Instruct')
"

# Silero VAD (輕量端點偵測)
git clone https://github.com/snakers4/silero-vad.git snakers4_silero-vad
```

## 環境變數

所有機敏資訊透過環境變數或 YAML 配置傳入，不要將 API Key 寫死在程式碼中。

| 變數 | 說明 | 範例 |
|------|------|------|
| `QWEN3_OMNI_API_URL` | vLLM-Omni API 端點 | `http://localhost:8901/v1` |
| `QWEN3_OMNI_MODEL` | 模型名稱或路徑 | `Qwen3-Omni-30B-A3B-Instruct` |
| `QWEATHER_API_HOST` | QWeather API host | `devapi.qweather.com` |
| `QWEATHER_API_KEY` | QWeather API Key | `your-api-key` |
| `VAD_MODEL_DIR` | Silero VAD 模型目錄 | `models/snakers4_silero-vad` |

在 `configs/xiaozhi_voice_e2e.yml` 中填入對應值，或建立 `.env` 檔案：

```bash
cp .env.example .env
# 編輯 .env 填入你的設定
```

## 安裝

### 1. 安裝 vLLM-Omni (WSL2)

```bash
wsl -d Ubuntu-22.04

cd ~/qwen3-omni
python3 -m venv venv
source venv/bin/activate

# 安裝 vLLM-Omni (含 Talker + Code2Wav 支援)
pip install vllm-omni

# 確認 CUDA
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 2. 部署 async_chunk stage config

本專案已附帶 `scripts/qwen3_omni_single_gpu_async.yaml`，複製到 WSL2 模型目錄即可：

```bash
# 從 Windows 複製到 WSL2
cp scripts/qwen3_omni_single_gpu_async.yaml ~/qwen3-omni/
```

### 3. 安裝 NAT + Voice Agent Plugin (Windows)

有兩種安裝方式。**方式 A** 最簡單；**方式 B** 適合需要修改 NAT 原始碼的情況。

#### 方式 A — pip install (推薦)

```bash
git clone -b qwen-omni https://github.com/tommywu052/nat-xiaozhi-voice-agent.git
cd nat-xiaozhi-voice-agent

python -m venv .venv
.venv\Scripts\activate    # Windows

pip install -e .
```

> `pip install -e .` 會自動安裝 `nvidia-nat[langchain]` 及所有依賴。

#### 方式 B — 從 NAT 原始碼安裝

```bash
git clone -b main https://github.com/NVIDIA/NeMo-Agent-Toolkit.git
cd NeMo-Agent-Toolkit
git submodule update --init --recursive

uv venv --python 3.12 --seed .venv
.venv\Scripts\activate

uv sync
uv pip install -e ".[langchain]"

# 安裝 Voice Agent Plugin
uv pip install -e /path/to/nat-xiaozhi-voice-agent
```

### 4. PyTorch 版本修正 (兩種方式皆需)

安裝過程中可能拉入不匹配的 PyTorch 版本，安裝完成後覆蓋為正確版本：

```bash
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --reinstall
```

## 啟動

### Step 1: 啟動 vLLM-Omni (WSL2)

使用專案附帶的啟動腳本：

```bash
wsl -d Ubuntu-22.04
bash scripts/start_vllm_omni_serve.sh
```

或手動啟動：

```bash
wsl -d Ubuntu-22.04
source ~/qwen3-omni/venv/bin/activate

vllm serve $QWEN3_OMNI_MODEL_PATH \
    --omni \
    --stage-configs-path ~/qwen3-omni/qwen3_omni_single_gpu_async.yaml \
    --port 8901 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --allowed-local-media-path / \
    --stage-init-timeout 600
```

> `--omni` + `--stage-configs-path` 啟用三階段 (Thinker + Talker + Code2Wav)。
> 缺少這兩個參數只會啟動 Thinker（純文字，無語音輸出）。

### Step 2: 設定 config

複製範本並填入你的環境設定：

```bash
cp .env.example .env
# 編輯 .env
```

或直接編輯 `configs/xiaozhi_voice_e2e.yml`：

```yaml
general:
  front_end:
    # vLLM-Omni 端點（改為你的 WSL2 IP 或 localhost）
    qwen3_omni_api_url: "http://localhost:8901/v1"
    qwen3_omni_model: "Qwen3-Omni-30B-A3B-Instruct"

    # VAD 模型路徑（相對或絕對）
    vad_model_dir: "models/snakers4_silero-vad"

    # 選配：天氣 API (QWeather)
    weather_api_host: "devapi.qweather.com"
    weather_api_key: ""   # 填入你的 QWeather API Key
```

> 查詢 WSL2 IP: `wsl -d Ubuntu-22.04 -- hostname -I`

### Step 3: 啟動 Voice Agent (Windows)

```bash
cd nat-xiaozhi-voice-agent
.venv\Scripts\activate

nat start xiaozhi_voice --config_file configs/xiaozhi_voice_e2e.yml
```

啟動成功後會看到：

```
Voice pipeline ready (VAD + E2E Qwen3-Omni async_chunk streaming + tool calling)
Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 客戶端連接

WebSocket 端點：`ws://<SERVER_IP>:8000/xiaozhi/v1/`

支援的客戶端：
- [py-xiaozhi](https://github.com/zhayujie/py-xiaozhi)（Python 桌面客戶端）
- [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32)（ESP32 硬體裝置）
- 任何相容小智 WebSocket 協議的客戶端

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查（回傳管線狀態、模式、連線數） |

健康檢查回傳範例：

```json
{
  "status": "ok",
  "connections": 1,
  "pipeline": {
    "mode": "e2e_streaming",
    "vad": true,
    "omni_e2e": true,
    "streaming": true
  }
}
```

## 內建工具

所有工具預設皆已註冊，啟動時不會報錯。E2E 模式下工具透過 Qwen3-Omni 原生 `<tool_call>` XML 標籤呼叫，由 `E2EToolExecutor` 執行。

| 工具 | 執行時需求 | 說明 |
|------|-----------|------|
| `current_datetime` | 無 | 查詢目前日期時間（Asia/Taipei 時區） |
| `get_weather` | QWeather API Key | 即時天氣 + 3 天預報（QWeather REST API） |
| `get_lunar` | 無 | 農曆日期、干支、生肖、節氣、黃曆宜忌（cnlunar） |
| `wiki_search` | 無 | 中文維基百科搜尋 |

### E2E Tool Calling 流程

```
1. 用戶語音 → vLLM-Omni stream_e2e() 開始串流
2. text stream 中偵測 <tool_call>{"name": "get_weather", "arguments": {"location": "台北"}}</tool_call>
3. 中斷音訊串流 → E2EToolExecutor 執行工具
4. 帶工具結果重新呼叫 stream_e2e() (text-only input)
5. 輸出語音回答（含工具結果）
```

## 效能參考 (RTX PRO 6000 Blackwell 98GB + Qwen3-Omni 30B-A3B)

| 階段 | 延遲 |
|------|------|
| VAD 端點偵測 | ~0.1s |
| Text TTFT (首文字) | ~1.3s |
| Audio TTFA (首音訊) | ~2.0s |
| Tool 偵測 + 執行 | ~1.5–3.0s |
| Tool followup 首音訊 | ~2.0s |
| **端到端（純對話）** | **~2.0–2.5s** |
| **端到端（含工具）** | **~4.0–5.5s** |

## 目錄結構

```
nat-xiaozhi-voice-agent/
├── configs/
│   └── xiaozhi_voice_e2e.yml              # E2E 統一配置檔
├── scripts/                                # vLLM-Omni 啟動腳本與 stage config
│   ├── start_vllm_omni_serve.sh           # E2E 模式啟動 (含 Talker + Code2Wav)
│   ├── start_vllm_serve.sh                # Thinker-only 模式啟動
│   └── qwen3_omni_single_gpu_async.yaml   # async_chunk 三階段並行配置
├── .env.example                            # 環境變數範本
├── src/nat_xiaozhi_voice/
│   ├── frontend/                       # NAT 前端插件（WebSocket 伺服器）
│   │   ├── config.py                   # Pydantic 配置（pipeline_mode, e2e_tool_calling 等）
│   │   ├── connection.py               # 單一連線處理（VAD → E2E streaming → Tool → Audio）
│   │   ├── plugin.py                   # NAT FrontEndBase 實作
│   │   ├── register.py                 # NAT 前端註冊
│   │   └── ws_server.py                # FastAPI WebSocket 伺服器
│   ├── pipeline/                       # 語音管線元件
│   │   ├── omni_e2e.py                 # Qwen3-Omni E2E 核心 (stream_e2e, transcribe, 對話歷史)
│   │   ├── tools.py                    # E2E 工具定義 + <tool_call> 解析 + E2EToolExecutor
│   │   └── vad.py                      # Silero ONNX VAD (語音端點偵測)
│   ├── utils/
│   │   ├── audio_codec.py              # Opus 編解碼
│   │   ├── audio_rate_controller.py    # 音訊播放速率控制
│   │   └── auth.py                     # HMAC 裝置驗證
│   └── workflow/
│       └── register.py                 # LangGraph Agent 定義（NAT 框架所需）
├── pyproject.toml
└── README.md
```

## 疑難排解

**Q: vLLM-Omni 啟動失敗 / OOM**
A: Qwen3-Omni 30B-A3B BF16 需要 ~79–98 GB VRAM。確認 GPU 記憶體足夠：
```bash
nvidia-smi
```

**Q: `ConnectionRefusedError` 連接 vLLM**
A: 確認 WSL2 IP 正確且 vLLM 已啟動：
```bash
# 查 WSL2 IP
wsl -d Ubuntu-22.04 -- hostname -I

# 測試連通
curl http://<WSL2_IP>:8901/v1/models
```

**Q: E2E 模式沒有音訊輸出**
A: 確認使用 `--omni` 和 `--stage-configs-path` 啟動 vLLM-Omni。普通 `vllm serve` 只有 Thinker（純文字），不會產生語音。

**Q: Tool Calling 未觸發**
A: 確認 `e2e_tool_calling: true` 已在 YAML 中啟用。工具定義會自動附加到 system prompt，Qwen3-Omni 原生支援 `<tool_call>` 格式。

**Q: E2E 音訊斷斷續續**
A: 可能是 GPU 記憶體不足導致推論變慢。調低 `kv_cache_memory_bytes` 或 `max_num_seqs` 參數。

**Q: `ModuleNotFoundError: No module named 'torch'`**
A: PyTorch 版本不匹配。強制重裝：
```bash
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --reinstall
```

**Q: `Input tag 'wiki_search' does not match any of the expected tags`**
A: NAT langchain 插件未安裝。方式 A 使用者重新執行 `pip install -e .`；方式 B 使用者執行：
```bash
cd NeMo-Agent-Toolkit
uv pip install -e ".[langchain]"
```

**Q: Port 8000/8901 already in use**
A: 先清除佔用的 process：
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# WSL2
lsof -ti:8901 | xargs -r kill -9
```

**Q: 重建 .venv 後 plugin 消失**
A: 方式 A 重新 `pip install -e .`。方式 B：
```bash
uv pip install -e ".[langchain]"
uv pip install -e /path/to/nat-xiaozhi-voice-agent
```

## 授權

MIT License
