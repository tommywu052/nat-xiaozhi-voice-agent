# Xiaozhi Voice Agent (NAT Plugin)

**English** | [繁體中文](README.md)

A real-time voice conversation Agent built on NVIDIA NeMo Agent Toolkit (NAT), compatible with [Xiaozhi ESP32](https://github.com/78/xiaozhi-esp32) / [py-xiaozhi](https://github.com/zhayujie/py-xiaozhi) client protocol.

The original Xiaozhi AI multi-process dual-system architecture has been fully migrated into **NAT custom components** — 15 Python modules, 0 lines of NAT source modified, 1 unified YAML config, independently installable via `pip install`.

---

## Changelog

| Date | Version | Summary |
|------|---------|---------|
| 2026-04-29 | **v3 — Nemotron-3-Nano-Omni** | Unified frontend LLM+VLM to [Nemotron-3-Nano-Omni](https://blogs.nvidia.com/blog/nemotron-3-nano-omni-multimodal-ai-agents/) 30B-A3B MoE (NVFP4). Disabled reasoning mode, reducing First Token latency from 10-14s to 0.2-0.6s. OpenClaw backend LLM also switched to Nano Omni (vLLM). Added 60-char hard reply truncation, TTS markdown cleanup, and enhanced Tavily web_search routing. |
| 2026-04-28 | v2 — Gemma4 E2B + OpenClaw | Frontend LLM switched to Gemma4 E2B (2B). Added OpenClaw dual-LLM architecture, async callback, Robot Camera Relay. |
| 2026-04-27 | v1 — Nemotron 3 Super | Initial version. Frontend LLM using Nemotron 3 Super (Ollama), basic voice pipeline + Tool Calling. |

---

### Demo

[![NAT Voice Agent Demo](docs/demo-thumbnail.png)](https://youtu.be/F58cTtn1T2I)

## Why Migrate to NAT?

### Four NAT Extension Points Used

| Extension | Purpose | Result |
|-----------|---------|--------|
| `@register_front_end` | Custom WebSocket server replacing NAT's default FastAPI | 100% compatible with original Xiaozhi protocol |
| `@register_function` | LangGraph StateGraph Agent replacing default tool_calling_agent | Agent and voice pipeline in same process, zero proxy latency |
| `entry_points` | pyproject.toml declares nat.components + nat.front_ends | `pip install` auto-registers; NAT upgrades don't overwrite custom code |
| `YAML _type` | Unified xiaozhi_voice.yml replaces 2 separate config files | Single file manages Front End + Pipeline + Agent + Tools |

### Migration Benefits

| Benefit | Description |
|---------|-------------|
| Reduced Latency | Removed HTTP proxy layer; agent calls are now in-process function calls, saving 50-100ms per turn |
| Simplified Ops | Service processes reduced from 5 to 3; config files consolidated from 2 to 1 YAML |
| Enhanced Observability | Phoenix traces the full chain: VAD → ASR → LLM → Tool → TTS (no more black boxes) |
| Extensibility | NAT's new tools (Code Exec, RAG, etc.) work out of the box; `pip install --upgrade nvidia-nat` won't break custom code |
| Unified Memory | Conversation memory natively managed by NAT Workflow with per-device-id persistence |
| Safe Rollback | `pip uninstall nat-xiaozhi-voice-agent` fully reverts; can run alongside the original system |

### Results

| Metric | Value |
|--------|-------|
| Python Modules | 15 |
| NAT Extension Points | 4 (front_end + function + entry_points + YAML _type) |
| NAT Source Modified | **0 lines** |
| Config Files | 1 unified YAML |
| Voice Pipeline Parameter Consistency | 7 key parameters 100% identical to original system |
| Client Modifications | **Zero** — ESP32 / py-xiaozhi migrate directly |

## Architecture

```
Voice Client (ESP32 / py-xiaozhi)         Robot Camera (camera_server.py)
    │  WebSocket (Opus audio + JSON)           │  WebSocket --relay
    ▼                                          ▼
┌──────────────────────────────────────────────────┐
│  Xiaozhi Voice Agent (NAT Plugin)   port 8000    │
│                                                  │
│  ┌─────┐  ┌─────┐  ┌────────────┐               │
│  │ VAD │→ │ ASR │→ │ LLM Agent  │──explain_scene │
│  │Silero│  │Fun  │  │ (LangGraph)│   ↕ Relay     │
│  │     │  │ASR  │  │  + Tools   │               │
│  └─────┘  └─────┘  └─────┬──────┘               │
│                           │                      │
│                     ┌─────▼──────┐               │
│                     │    TTS     │               │
│                     │ (Edge TTS) │               │
│                     └────────────┘               │
└──────────────────────────────────────────────────┘
```

**Voice Pipeline:**
- **VAD** — Silero VAD (ONNX), voice activity detection
- **ASR** — FunASR SenseVoiceSmall, supports Chinese / English / Japanese / Cantonese
- **LLM+VLM** — Nemotron-3-Nano-Omni (30B-A3B MoE), unified multimodal model (text+vision+audio+video), via vLLM OpenAI-compatible API
- **TTS** — Microsoft Edge TTS (free cloud) or CosyVoice (local)

**Features:**
- LangGraph-based ReAct Agent with Tool Calling
- Per-device persistent conversation memory (SQLite)
- History compression: auto-summarizes old conversations when threshold is exceeded
- Streaming TTS: synthesizes speech as LLM generates tokens, reducing first-audio latency

## OpenClaw Integration — Unified LLM Architecture & Async Callback

This project supports deep integration with [OpenClaw](https://docs.openclaw.ai/), enabling a "voice frontend + AI agent backend" unified LLM collaboration architecture. Users issue complex commands to Xiaozhi via voice; Xiaozhi automatically delegates them to OpenClaw for background execution and proactively announces the results via voice upon completion.

> **v3 Change**: Both frontend and OpenClaw backend now use Nemotron-3-Nano-Omni (vLLM). Ollama is no longer required.

### Unified LLM Architecture

```
User Voice
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  NAT Xiaozhi Voice Agent                                     │
│                                                              │
│  ┌─────┐  ┌─────┐  ┌──────────────────────────────────────┐ │
│  │ VAD │→ │ ASR │→ │ Nemotron-3-Nano-Omni — Frontend LLM+VLM│
│  │     │  │     │  │  vLLM localhost:8880                  │ │
│  └─────┘  └─────┘  │  • 30B-A3B MoE (NVFP4 quantization)  │ │
│                     │  • enable_thinking=false (low latency)│ │
│                     │  • Native multimodal (vision/audio/video)│
│                     │  • 60-char hard truncation + streaming TTS│
│                     └──────────┬───────────────────────────┘ │
│                                │                             │
│              ┌─────────────────┼─────────────────┐           │
│              │ Simple tools    │ Complex tasks    │           │
│              ▼                 ▼                  │           │
│         web_search       ┌─────────┐             │           │
│         wiki_search      │openclaw │ ──(CLI)──→  │           │
│         current_datetime │  tool   │             │           │
│         explain_scene    └────┬────┘             │           │
│                               │                  │           │
│                     ┌─────────▼──────────┐       │           │
│                     │  /api/speak        │←──────┘           │
│                     │  Async callback TTS│                   │
│                     └────────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
                                │
                    openclaw agent CLI
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  OpenClaw Agent                                              │
│  Nemotron-3-Nano-Omni — Backend LLM (unified model)         │
│  vLLM localhost:8880 (shared vLLM instance)                  │
│                                                              │
│  • Deep web research (web_search → web_fetch → analysis)     │
│  • Cross-channel delivery (WhatsApp, LINE, Telegram…)        │
│  • Cron job scheduling                                       │
│  • Long-form report generation                               │
└──────────────────────────────────────────────────────────────┘
```

### Why Unified LLM?

In v2, the frontend used Gemma4 E2B (2B) and the backend used Nemotron 3 Super (123.6B Q4), requiring two separate inference services (vLLM + Ollama). v3 unifies on Nemotron-3-Nano-Omni with the following advantages:

| | v2 (Dual LLM) | v3 (Unified LLM) |
|---|---|---|
| **Frontend Model** | Gemma4 E2B (2B) | Nemotron-3-Nano-Omni 30B-A3B |
| **Backend Model** | Nemotron 3 Super 123.6B (Q4) | Nemotron-3-Nano-Omni 30B-A3B (shared) |
| **Inference Service** | vLLM + Ollama dual services | Single vLLM instance |
| **Total GPU Usage** | ~20 GB + ~84 GB ≈ 104 GB | ~20 GB (NVFP4 quantization) |
| **Multimodal** | Frontend text-only, VLM needs separate model | Native text+image+audio+video |
| **Tool Calling** | Frontend only | Both frontend and backend |
| **Maintenance** | High (two configs, two services) | Low (single vLLM config) |

A single Nemotron-3-Nano-Omni runs on NVIDIA GB10's 128GB unified memory with NVFP4 quantization, using only ~20 GB while serving both frontend voice interaction and OpenClaw backend tasks.

### Async Callback Mechanism

OpenClaw tasks typically take several minutes (deep search, multi-turn tool calls). The system uses async callbacks to keep the voice experience smooth:

```
1. User: "Research AI applications in manufacturing"
2. Xiaozhi: "OK, processing in background. I'll notify you when done."  ← Instant reply (<2s)
3. User can continue chatting with Xiaozhi about other topics
4. [2-5 minutes later] OpenClaw completes research
5. Xiaozhi proactively announces: "OpenClaw is done. AI is profoundly transforming manufacturing…"
```

**Key Endpoint:** `POST /api/speak`

```bash
# External systems can push proactive voice notifications to connected clients
curl -X POST http://localhost:8000/api/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Your background task is complete!", "device_id": ""}'
```

| Parameter | Description |
|-----------|-------------|
| `text` | Text to announce (synthesized via TTS and played back) |
| `device_id` | Target device (empty = broadcast to all connected clients) |

**Automatic Routing Logic:** The `openclaw` tool has built-in keyword detection for automatic sync/async mode selection:

| Mode | Trigger Keywords | Behavior |
|------|-----------------|----------|
| **Sync** | General queries | Waits for OpenClaw reply (≤120s), responds via voice directly |
| **Async** | research, analyze, WhatsApp, schedule, report… | Immediately confirms, waits in background, POSTs to `/api/speak` on completion |

### OpenClaw Setup Steps

**Step 1 — Install OpenClaw:**

```bash
curl -fsSL https://get.openclaw.ai | sh
openclaw onboard --auth-choice ollama
openclaw configure
```

**Step 2 — Set up vLLM + Nemotron-3-Nano-Omni (unified frontend + backend model):**

```bash
# Option A: Use the startup script (recommended)
./scripts/start-nano-omni-docker.sh          # NVFP4 (default, Blackwell GPU)
./scripts/start-nano-omni-docker.sh --bf16   # BF16 (H100 / ≥64GB VRAM)

# Option B: Manual Docker launch
docker run -d --name vllm-nano-omni \
  --runtime nvidia --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8880:8880 \
  vllm/vllm-openai:latest \
    --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
    --served-model-name nemotron \
    --trust-remote-code \
    --host 0.0.0.0 --port 8880 \
    --max-model-len 131072 --max-num-seqs 8 \
    --kv-cache-dtype fp8 --moe-backend cutlass \
    --gpu-memory-utilization 0.85 \
    --media-io-kwargs '{"video":{"num_frames":512,"fps":1}}' \
    --video-pruning-rate 0.5 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

> **Note**: NVFP4 quantization requires Blackwell architecture GPU (GB10 / RTX PRO 6000).
> For H100, use BF16 or FP8 variants instead.
> See [NVIDIA vLLM Cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano-Omni/vllm_cookbook.ipynb) for detailed deployment guide.

**Step 3 — Configure OpenClaw to use vLLM (Nano Omni):**

Edit `~/.openclaw/openclaw.json` to add the vLLM provider and set it as default:

```json
{
  "agents": {
    "defaults": {
      "model": { "primary": "vllm/nemotron" }
    }
  },
  "models": {
    "providers": {
      "vllm": {
        "api": "openai",
        "apiKey": "not-needed",
        "baseUrl": "http://127.0.0.1:8880/v1",
        "models": [{
          "id": "nemotron",
          "name": "nemotron",
          "reasoning": false,
          "input": ["text", "image"],
          "contextWindow": 131072,
          "maxTokens": 8192,
          "compat": { "supportsTools": true }
        }]
      }
    },
    "mode": "merge"
  }
}
```

**Step 4 — Enable the openclaw tool in `xiaozhi_voice.yml`:**

```yaml
functions:
  openclaw:
    _type: openclaw
    session_id: "voice-bridge"
    sync_timeout: 120    # Max wait for sync mode (seconds)
    async_timeout: 600   # Max wait for async mode (seconds)

  voice_agent:
    tool_names:
      - current_datetime
      - wiki_search
      - web_search
      - explain_scene
      - openclaw          # Add the openclaw tool
```

**Step 5 — Start NAT Voice Agent:**

```bash
TAVILY_API_KEY="tvly-YOUR_KEY" ASR_DEVICE=cpu \
  nat start xiaozhi_voice --config_file configs/xiaozhi_voice.yml
```

> **Note**: `ASR_DEVICE=cpu` prevents FunASR from consuming GPU memory, ensuring vLLM has sufficient VRAM.

On successful startup, logs will show:
```
openclaw_delegate configured: session=voice-bridge sync=120s async=600s
openclaw tool registered (session=voice-bridge, sync=120s, async=600s)
```

### OpenClaw-Related Files

| File | Description |
|------|-------------|
| `src/nat_xiaozhi_voice/tools/openclaw_delegate.py` | Core delegation logic (sync/async routing, `/api/speak` callback, summarization) |
| `src/nat_xiaozhi_voice/tools/openclaw_register.py` | NAT tool registration (`@register_function`, empty message fallback) |
| `src/nat_xiaozhi_voice/frontend/ws_server.py` | `/api/speak` HTTP endpoint definition |
| `src/nat_xiaozhi_voice/frontend/connection.py` | `speak()` method (proactive TTS push, abort-aware, queue waiting) |

---

## Prerequisites

### System

- Linux (x86_64 or aarch64, verified on DGX Spark aarch64)
- Python 3.11 / 3.12 / 3.13
- [NVIDIA API Key](https://build.nvidia.com/) (for LLM / VLM inference)

### System Packages (not needed for Docker)

```bash
sudo apt-get install -y libopus-dev libopus0 libsndfile1-dev
```

### Model Files (auto-downloaded in Docker)

For native installation, download two models into the `models/` directory:

```bash
# 1. Silero VAD
cd models/
git clone https://github.com/snakers4/silero-vad.git snakers4_silero-vad

# 2. FunASR SenseVoiceSmall
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/SenseVoiceSmall', local_dir='models/SenseVoiceSmall')
"
```

## Installation

Three installation methods. **Docker** is the fastest (one command); **Method A** for native install; **Method B** for developers.

---

### Docker (Fastest)

Requires Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
git clone https://github.com/tommywu052/nat-xiaozhi-voice-agent.git
cd nat-xiaozhi-voice-agent

# Create .env file with API keys
echo 'NVIDIA_API_KEY=nvapi-YOUR_KEY' > .env
echo 'TAVILY_API_KEY=tvly-YOUR_KEY' >> .env   # optional

# Build and start (first run downloads models, ~5-10 min)
docker compose up -d --build
```

WebSocket endpoint after startup: `ws://localhost:8000/xiaozhi/v1/`

```bash
# View logs
docker compose logs -f

# Stop
docker compose down
```

> The Docker image includes CUDA runtime, PyTorch, ASR/VAD models — **no additional dependencies needed**.

**Enable explain_scene Camera:**

Option A — **Remote Robot Camera via Relay (recommended):** No Docker changes needed. Set `relay_enabled: true` in the mounted YAML config, then connect the robot with `camera_server.py --relay ws://NAT_SERVER:8000/ws/robot`.

Option B — **Local USB Webcam:** The container cannot access host USB cameras by default. Uncomment the `devices` section in `docker-compose.yml`:

```yaml
    devices:
      - /dev/video0:/dev/video0
```

Verify the camera device path:

```bash
ls /dev/video*
# Usually /dev/video0 is the first USB camera
```

> If no camera is connected and no robot is relayed, `explain_scene` returns a friendly error message when called.

---

### Method A — pip install (Native)

NAT is published on PyPI (`nvidia-nat`); no need to clone the NAT repo.

```bash
git clone https://github.com/tommywu052/nat-xiaozhi-voice-agent.git
cd nat-xiaozhi-voice-agent

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

> `pip install -e .` automatically installs `nvidia-nat[langchain]` and all dependencies (LangChain, LangGraph, Tavily, etc.).

---

### Method B — Install from NAT Source

For developing / debugging NAT core, or using the latest git main features.

```bash
git clone -b main https://github.com/NVIDIA/NeMo-Agent-Toolkit.git
cd NeMo-Agent-Toolkit
git submodule update --init --recursive

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create virtual environment
uv venv --python 3.12 --seed .venv
source .venv/bin/activate

# Install NAT core + langchain plugin
uv sync
uv pip install -e ".[langchain]"

# Install Voice Agent Plugin
uv pip install -e /path/to/nat-xiaozhi-voice-agent
```

---

### PyTorch Version Fix (Required for Method A & B)

`funasr` may pull in an incompatible PyTorch version.
**After installation**, run the following to override with the correct version:

**With NVIDIA GPU + CUDA (DGX Spark / Jetson, etc.) — Recommended:**

```bash
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --reinstall
```

> ASR on GPU is ~8x faster than CPU (59ms vs 460ms for 5-second audio).
> DGX Spark's NVIDIA GB10 has 120GB VRAM; SenseVoiceSmall only uses a few hundred MB.

**Without GPU or CUDA:**

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Configure NAT Timezone (Recommended)

NAT's `current_datetime` tool defaults to UTC. If your system timezone is correctly set (e.g., `Asia/Taipei`), run:

```bash
mkdir -p ~/.config/nat
echo '{"fallback_timezone": "system"}' > ~/.config/nat/config.json
```

Verify system timezone:

```bash
timedatectl | grep "Time zone"
# Expected: Time zone: Asia/Taipei (CST, +0800)
```

### Configure

Edit `configs/xiaozhi_voice.yml`:

```yaml
# Required changes:
llms:
  main_llm:
    api_key: "nvapi-YOUR_NVIDIA_API_KEY"    # Replace with your NVIDIA API Key

general:
  front_end:
    vad_model_dir: "/absolute/path/to/models/snakers4_silero-vad"
    asr_model_dir: "/absolute/path/to/models/SenseVoiceSmall"

# Optional: enable web_search (Tavily)
functions:
  web_search:
    api_key: "tvly-YOUR_TAVILY_KEY"   # Or set via TAVILY_API_KEY env variable
```

## Launch

```bash
source .venv/bin/activate
export NVIDIA_API_KEY="nvapi-YOUR_KEY"

nat start xiaozhi_voice \
    --config_file /path/to/nat-xiaozhi-voice-agent/configs/xiaozhi_voice.yml
```

On successful startup you'll see:

```
Voice pipeline ready (VAD + ASR + TTS)
Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
LLM warm-up done in 0.77s
```

## Client Connection

| Endpoint | Purpose |
|----------|---------|
| `ws://<SERVER_IP>:8000/xiaozhi/v1/` | Voice client (ESP32 / py-xiaozhi) |
| `ws://<SERVER_IP>:8000/ws/robot` | Robot camera relay (when relay_enabled: true) |

Supported voice clients:
- **client/py-xiaozhi-ws.py** (built-in test client, see below)
- [py-xiaozhi](https://github.com/zhayujie/py-xiaozhi) (Python desktop client)
- [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) (ESP32 hardware device)
- Any client compatible with the Xiaozhi WebSocket protocol

### Built-in Test Client — client/py-xiaozhi-ws.py

The project includes `client/py-xiaozhi-ws.py` for quick desktop voice testing. Hold spacebar to talk, release to send, ESC to quit.

**Additional dependencies:**

```bash
pip install pyaudio pynput pyserial
```

> `pyaudio` requires system PortAudio: `sudo apt-get install -y portaudio19-dev`

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `XIAOZHI_DEVICE_ID` | *(required)* | Device ID for client identification and memory binding |
| `XIAOZHI_WS_URL` | `ws://localhost:8000/xiaozhi/v1/` | Voice Agent WebSocket endpoint |
| `XIAOZHI_CLIENT_ID` | `py-xiaozhi-ws-client` | Client identifier |
| `EYE_SERIAL_PORT` | `COM10` | ESP32 eye module serial port (ignore if no hardware) |

**Usage:**

```bash
# Basic (using defaults)
python client/py-xiaozhi-ws.py

# Custom device ID and server
XIAOZHI_DEVICE_ID="your-device-id" \
XIAOZHI_WS_URL="ws://192.168.1.100:8000/xiaozhi/v1/" \
python client/py-xiaozhi-ws.py
```

**Controls:**

| Key | Function |
|-----|----------|
| Hold Spacebar | Start recording |
| Release Spacebar | Stop recording and send |
| Number keys 0-9 | Manually switch ESP32 eye design |
| h / a / s / u / c / n | Emotion shortcuts (happy / angry / sad / surprised / confused / neutral) |
| ESC | Exit |

**Features:**
- Real-time Opus-encoded audio send / receive
- ESP32 eye expression module support (Serial, auto-skipped if no hardware)
- Auto-handles TTS interruption (press spacebar to abort)

## Robot Camera Relay

When the Robot (e.g., ESP32 device, Raspberry Pi, Windows PC) has a dynamic IP or sits behind NAT, use the built-in WebSocket Relay so the Robot **actively connects into** the NAT server — no need to know the Robot's IP.

### Architecture

```
Robot (dynamic IP)                         NAT Server (fixed/known IP)
┌──────────────────┐                      ┌─────────────────────────────────────┐
│ camera_server.py │                      │  Xiaozhi Voice Agent (port 8000)   │
│   --relay ws://  │──WebSocket connect──→│    /xiaozhi/v1/  (voice WS)        │
│   NAT:8000/ws/   │                      │    /ws/robot     (camera relay)    │
│   robot          │  ←──capture request──│    /health       (health check)    │
│                  │  ──image response──→ │                                     │
└──────────────────┘                      └─────────────────────────────────────┘
```

### Setup Steps

**Step 1 — Enable relay in NAT config (YAML):**

```yaml
general:
  front_end:
    relay_enabled: true    # Enables the /ws/robot WebSocket endpoint
```

**Step 2 — Start NAT server (single command; relay starts automatically):**

```bash
nat start xiaozhi_voice --config_file configs/xiaozhi_voice.yml
```

**Step 3 — Connect the Robot:**

```bash
# On the Robot machine, install dependencies
pip install opencv-python websockets

# Connect to the NAT server's relay
python client/camera_server.py --relay ws://NAT_SERVER_IP:8000/ws/robot
```

### Verify Connection

```bash
# Check health endpoint — robot_connected should be true
curl http://localhost:8000/health
# {"status":"ok","connections":0,"pipeline":{...},"robot_connected":true}
```

### camera_server.py Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Relay** | `--relay ws://NAT:8000/ws/robot` | Recommended. Robot connects into NAT; dynamic IP is fine |
| **HTTP** | `--port 9903` | Direct connection. NAT must know Robot IP |
| **MCP stdio** | `--mcp` | Legacy Xiaozhi Cloud compatibility mode |

### Performance

The relay architecture has zero negative performance impact. Camera capture via the in-process WebSocket bridge takes only **0.02-0.03s**, faster than direct HTTP (0.06s).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (returns pipeline status + robot_connected) |
| GET | `/api/memory` | List all devices with conversation memory |
| DELETE | `/api/memory/{device_id}` | Clear conversation memory for a device |
| DELETE | `/api/memory` | Clear all conversation memory |
| POST | `/api/speak` | Proactive voice push (OpenClaw async callback / external notifications) |
| WS | `/xiaozhi/v1/` | Voice client WebSocket (ESP32 / py-xiaozhi) |
| WS | `/ws/robot` | Robot camera relay WebSocket (when relay_enabled: true) |

## Built-in Tools

All tools are registered at startup **without errors**. Resources are only accessed when the LLM invokes the tool; if unavailable, a friendly error message is returned instead of a crash.

| Tool | Runtime Requirement | Description |
|------|-------------------|-------------|
| `current_datetime` | None | Query current date/time (uses system timezone) |
| `wiki_search` | None | Wikipedia search |
| `web_search` | Tavily API Key | Real-time web search (news, weather, etc.); returns error if no key |
| `explain_scene` | Camera (relay / HTTP / USB) | Describe camera view via VLM; supports remote robot camera relay |
| `openclaw` | OpenClaw + vLLM | Delegate complex tasks to OpenClaw Agent (research, WhatsApp, scheduling, reports) |

### explain_scene Configuration

`explain_scene` supports three camera sources (by priority):

| Priority | Source | Use Case |
|:---:|--------|----------|
| 1 | **Built-in Relay** — Robot connects via WebSocket to NAT | Dynamic Robot IP / cross-network (recommended) |
| 2 | **Remote HTTP** — NAT connects directly to Robot camera server | Fixed Robot IP, same subnet |
| 3 | **Local USB** — NAT server's local webcam | Development / all-in-one deployment |

**Mode 1 — Built-in Relay (recommended):**

```yaml
general:
  front_end:
    relay_enabled: true   # Enable WebSocket relay; Robot connects to ws://host:port/ws/robot

functions:
  explain_scene:
    _type: explain_scene
    llm_name: main_llm
    vlm_model: "google/gemma-4-31b-it"
```

On the Robot side:

```bash
python client/camera_server.py --relay ws://NAT_SERVER_IP:8000/ws/robot
```

See the "Robot Camera Relay" section above for full details.

**Mode 2 — Remote HTTP (direct connection):**

```yaml
functions:
  explain_scene:
    _type: explain_scene
    remote_camera_url: "http://ROBOT_IP:9903"
    llm_name: main_llm
    vlm_model: "google/gemma-4-31b-it"
```

**Mode 3 — Local USB webcam:**

```yaml
functions:
  explain_scene:
    _type: explain_scene
    camera_index: 0
    llm_name: main_llm
    vlm_model: "google/gemma-4-31b-it"
```

**Available NVIDIA NIM Vision Models:**

| Model | Chinese Support | Accuracy | Response Time (warm) |
|-------|----------------|----------|---------------------|
| `google/gemma-4-31b-it` | Excellent (Traditional Chinese) | High (correctly identified RTX 5090) | ~2.5s |
| `meta/llama-3.2-11b-vision-instruct` | Poor (often replies in English) | Medium (misidentified model) | ~8s |
| `meta/llama-3.2-90b-vision-instruct` | Medium | High | ~15s |
| `microsoft/phi-4-multimodal-instruct` | Medium | Medium | TBD |

> **Recommended:** `google/gemma-4-31b-it` — best Chinese capability and recognition accuracy. First call has ~35s cold start, then stable at 2-3 seconds.

## LLM Model Recommendations

| Model | Chat TTFT | Tool TTFT | Tool Calling | Chinese Quality | Use Case |
|-------|-----------|-----------|-------------|----------------|----------|
| `qwen/qwen3-next-80b-a3b-instruct` | ~0.44s | ~1.5s | Excellent (85% <2s) | Excellent | **Recommended** — accurate tool judgment, strong comprehension |
| `meta/llama-3.1-8b-instruct` | ~0.17s | ~1.0s | Fair (frequent misjudgment) | Fair | Low-latency chat only |
| `meta/llama-3.3-70b-instruct` | ~0.6s | ~12s | Full support | Good | Complex tasks, tool-heavy |
| `nvidia/llama-3.1-nemotron-nano-8b-v1` | ~1.8s | N/A | Not supported | Fair | Chat only (not recommended) |

> **Recommended:** `qwen/qwen3-next-80b-a3b-instruct` for voice agents — far superior to 8B models in tool call judgment and semantic understanding.

## Performance Reference

### v3 — Nemotron-3-Nano-Omni (NVFP4, DGX Spark aarch64)

> Test environment: NVIDIA DGX Spark (GB10, 128GB unified memory), vLLM OpenAI-compatible API, enable_thinking=false

| Stage | Latency | Notes |
|-------|---------|-------|
| VAD + ASR (SenseVoiceSmall, CPU) | ~0.3-0.5s | ASR runs on CPU to save VRAM |
| LLM Chat TTFT | ~0.2-0.6s | Significantly improved after disabling reasoning mode |
| LLM + Tool Calling TTFT | ~0.8-2.5s | Varies by tool complexity |
| web_search (Tavily) | ~2-4s | Includes external API call |
| current_datetime | ~0.5s | Local tool, very fast |
| EdgeTTS Synthesis | ~0.5-1.0s | |
| History Compression (>10 turns) | +1-3s | Summarizing old conversations adds one LLM call |
| **End-to-end (chat only)** | **~1.0-1.5s** | |
| **End-to-end (with tools)** | **~2.0-5.0s** | |

### v3 Conversation Test Results (18 Voice Turns)

| Test Item | Result | Notes |
|-----------|--------|-------|
| Greetings / Small Talk | ✅ Pass | No tools triggered, Traditional Chinese replies |
| Date/Time Queries | ✅ Pass | Correctly invoked `current_datetime` |
| Weather Queries | ✅ Pass | Correctly invoked `web_search`, returned live weather |
| News Search | ✅ Pass | Correctly invoked `web_search` |
| Recommendations / Suggestions | ✅ Pass | Correctly invoked `web_search` |
| OpenClaw Deep Research | ✅ Pass | Async delegation successful, voice callback on completion |
| Language Consistency | ⚠️ Occasional | Occasional Simplified Chinese or English safety refusals |
| Reply Length Control | ✅ Pass | 60-char hard truncation ensures concise voice output |
| explain_scene Routing | ⚠️ Needs Improvement | Medical questions occasionally misrouted to explain_scene |

### v3 Key Optimizations

| Optimization | Approach | Effect |
|-------------|----------|--------|
| First Token Latency | `enable_thinking=false` | 10-14s → 0.2-0.6s |
| Reply Length Overflow | `max_tokens=200` + frontend 60-char hard truncation | TTS output is concise |
| Tool Call Truncation | Increased `max_tokens` to 200 | Tool JSON no longer truncated |
| TTS Noise | Enhanced `_clean_for_tts` (strips XML/markdown/lists) | Clean voice output |
| Weather Not Triggering Search | Fixed `max_tokens` + enhanced prompt routing rules | Reliably triggers `web_search` |
| OpenClaw Backend Unified | `~/.openclaw/openclaw.json` switched to vLLM | Ollama no longer required |

### Legacy Performance Reference (NIM API + qwen3-next-80b)

| Stage | Latency |
|-------|---------|
| VAD + ASR (SenseVoiceSmall) | ~0.4s |
| LLM Chat TTFT | ~0.4s |
| LLM + Lightweight Tool TTFT | ~1.5s |
| EdgeTTS Synthesis | ~0.5-1.0s |
| VLM explain_scene (warm) | ~2.5s |
| **End-to-end (chat only)** | **~1.0-1.5s** |

## Directory Structure

```
nat-xiaozhi-voice-agent/
├── configs/
│   └── xiaozhi_voice.yml          # Unified NAT config (incl. relay_enabled)
├── client/                         # Robot / desktop client scripts
│   ├── camera_server.py            # Robot camera server (relay / HTTP / MCP modes)
│   └── py-xiaozhi-ws.py            # Desktop test client (push-to-talk)
├── models/                         # Model files (download required)
│   ├── snakers4_silero-vad/        # Silero VAD ONNX model
│   └── SenseVoiceSmall/            # FunASR speech recognition model
├── src/nat_xiaozhi_voice/
│   ├── frontend/                   # NAT front-end plugin (WebSocket server)
│   │   ├── config.py               # Pydantic config definition (incl. relay_enabled)
│   │   ├── connection.py           # Per-connection handler (VAD→ASR→LLM→TTS)
│   │   ├── plugin.py               # NAT FrontEndBase implementation
│   │   ├── register.py             # NAT front-end registration
│   │   └── ws_server.py            # FastAPI WebSocket server + Robot Camera Relay
│   ├── pipeline/                   # Voice pipeline components
│   │   ├── asr.py                  # FunASR speech recognition
│   │   ├── tts.py                  # Edge TTS / CosyVoice synthesis
│   │   └── vad.py                  # Silero VAD voice activity detection
│   ├── tools/                      # Custom NAT tools
│   │   ├── register.py             # explain_scene registration (llm_name reference)
│   │   ├── vlm_camera.py           # VLM camera tool (relay / HTTP / local)
│   │   ├── openclaw_delegate.py    # OpenClaw delegation core (sync/async routing, /api/speak callback)
│   │   └── openclaw_register.py    # OpenClaw NAT tool registration (with empty message fallback)
│   ├── utils/
│   │   ├── audio_codec.py          # Opus encode/decode
│   │   ├── audio_rate_controller.py # Audio rate control
│   │   └── auth.py                 # JWT authentication
│   └── workflow/
│       └── register.py             # LangGraph Agent definition (with memory compression)
├── mcp_ws_relay.py                 # Standalone relay (backup, not needed with relay_enabled)
├── test_vlm.py                     # VLM vision model test script
├── Dockerfile                      # Docker container definition
├── docker-compose.yml              # One-command launch config
├── pyproject.toml
└── README.md
```

## Troubleshooting

**Q: `ModuleNotFoundError: No module named 'torch'` or `libc10_cuda.so: cannot open shared object`**
A: PyTorch was installed with CUDA but the system lacks CUDA drivers. Force install CPU version:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Q: `Input tag 'wiki_search' does not match any of the expected tags`**
A: NAT langchain plugin not installed. Method A users: re-run `pip install -e .`; Method B users:
```bash
cd NeMo-Agent-Toolkit
uv pip install -e ".[langchain]"
```

**Q: `LLM 'main_llm' not found` (explain_scene startup failure)**
A: When explain_scene uses `llm_name` to reference an LLM, ensure the `llms` section in YAML has a matching definition.

**Q: `Cannot open USB webcam (index=0)`**
A: Without a USB webcam, `explain_scene` won't crash — it returns a friendly "cannot open camera" message when called.

**Q: `current_datetime` returns UTC instead of local time**
A: Set NAT timezone to system timezone:
```bash
mkdir -p ~/.config/nat
echo '{"fallback_timezone": "system"}' > ~/.config/nat/config.json
```

**Q: `Error code: 400 - max_completion_tokens`**
A: Some NVIDIA NIM models don't support the `max_tokens` parameter. Remove it from the LLM section in config.

**Q: Port 8000 already in use**
A: Kill the occupying process:
```bash
lsof -ti:8000 | xargs -r kill -9
```

**Q: LLM responses are slow (>10s)**
A: Possible causes:
1. Model too large (e.g., 122B) — switch to `qwen/qwen3-next-80b-a3b-instruct`
2. Tool returns too much data — reduce `doc_content_chars_max`
3. Model repeatedly calls the same tool — add restrictions in system prompt

**Q: Plugins missing after rebuilding .venv**
A: Method A: re-run `pip install -e .`. Method B: `uv sync` resets the virtual environment; reinstall:
```bash
uv pip install -e ".[langchain]"
uv pip install -e /path/to/nat-xiaozhi-voice-agent
```
Both methods require re-running the PyTorch version fix step.
