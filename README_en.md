# Xiaozhi Voice Agent (NAT Plugin)

**English** | [繁體中文](README.md)

A real-time voice conversation Agent built on NVIDIA NeMo Agent Toolkit (NAT), compatible with [Xiaozhi ESP32](https://github.com/78/xiaozhi-esp32) / [py-xiaozhi](https://github.com/zhayujie/py-xiaozhi) client protocol.

The original Xiaozhi AI multi-process dual-system architecture has been fully migrated into **NAT custom components** — 15 Python modules, 0 lines of NAT source modified, 1 unified YAML config, independently installable via `pip install`.

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
Client (ESP32 / py-xiaozhi)
    │  WebSocket (Opus audio + JSON control)
    ▼
┌─────────────────────────────────────┐
│  Xiaozhi Voice Agent (NAT Plugin)   │
│                                     │
│  ┌─────┐  ┌─────┐  ┌────────────┐  │
│  │ VAD │→ │ ASR │→ │ LLM Agent  │  │
│  │Silero│  │Fun  │  │ (LangGraph)│  │
│  │     │  │ASR  │  │  + Tools   │  │
│  └─────┘  └─────┘  └─────┬──────┘  │
│                           │         │
│                     ┌─────▼──────┐  │
│                     │    TTS     │  │
│                     │ (Edge TTS) │  │
│                     └────────────┘  │
└─────────────────────────────────────┘
```

**Voice Pipeline:**
- **VAD** — Silero VAD (ONNX), voice activity detection
- **ASR** — FunASR SenseVoiceSmall, supports Chinese / English / Japanese / Cantonese
- **LLM** — Via NVIDIA NIM API (any OpenAI-compatible model)
- **TTS** — Microsoft Edge TTS (free cloud) or CosyVoice (local)

**Features:**
- LangGraph-based ReAct Agent with Tool Calling
- Per-device persistent conversation memory (SQLite)
- History compression: auto-summarizes old conversations when threshold is exceeded
- Streaming TTS: synthesizes speech as LLM generates tokens, reducing first-audio latency

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

**Enable USB Webcam (explain_scene tool):**

The container cannot access host USB cameras by default. To enable the `explain_scene` tool, uncomment the `devices` section in `docker-compose.yml`:

```yaml
    devices:
      - /dev/video0:/dev/video0
```

Verify the camera device path:

```bash
ls /dev/video*
# Usually /dev/video0 is the first USB camera
```

> If no camera is connected, no configuration needed. `explain_scene` returns a friendly error message when called without a camera.

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

WebSocket endpoint: `ws://<SERVER_IP>:8000/xiaozhi/v1/`

Supported clients:
- **py-xiaozhi-ws.py** (built-in test client, see below)
- [py-xiaozhi](https://github.com/zhayujie/py-xiaozhi) (Python desktop client)
- [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) (ESP32 hardware device)
- Any client compatible with the Xiaozhi WebSocket protocol

### Built-in Test Client — py-xiaozhi-ws.py

The project includes `py-xiaozhi-ws.py` for quick desktop voice testing. Hold spacebar to talk, release to send, ESC to quit.

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
python py-xiaozhi-ws.py

# Custom device ID and server
XIAOZHI_DEVICE_ID="your-device-id" \
XIAOZHI_WS_URL="ws://192.168.1.100:8000/xiaozhi/v1/" \
python py-xiaozhi-ws.py
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

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (returns pipeline status) |
| GET | `/api/memory` | List all devices with conversation memory |
| DELETE | `/api/memory/{device_id}` | Clear conversation memory for a device |
| DELETE | `/api/memory` | Clear all conversation memory |

## Built-in Tools

All tools are registered at startup **without errors**. Resources are only accessed when the LLM invokes the tool; if unavailable, a friendly error message is returned instead of a crash.

| Tool | Runtime Requirement | Description |
|------|-------------------|-------------|
| `current_datetime` | None | Query current date/time (uses system timezone) |
| `wiki_search` | None | Wikipedia search |
| `web_search` | Tavily API Key | Real-time web search (news, weather, etc.); returns error if no key |
| `explain_scene` | USB webcam | Describe camera view via VLM; returns "cannot open camera" if no webcam |

### explain_scene Configuration

`explain_scene` references an LLM defined in YAML via `llm_name`, sharing the NVIDIA NIM endpoint and API key. Only the multimodal vision model needs to be specified separately:

```yaml
functions:
  explain_scene:
    _type: explain_scene
    camera_index: 0
    llm_name: main_llm                      # Reuse main_llm's base_url and api_key
    vlm_model: "google/gemma-4-31b-it"      # Multimodal vision model
```

Standalone configuration (without LLM reference) is also supported:

```yaml
functions:
  explain_scene:
    _type: explain_scene
    camera_index: 0
    vlm_base_url: "https://integrate.api.nvidia.com/v1"
    vlm_api_key: "nvapi-YOUR_KEY"
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

## Performance Reference (DGX Spark aarch64 + qwen3-next-80b)

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
│   └── xiaozhi_voice.yml          # Unified NAT config
├── models/                         # Model files (download required)
│   ├── snakers4_silero-vad/        # Silero VAD ONNX model
│   └── SenseVoiceSmall/            # FunASR speech recognition model
├── src/nat_xiaozhi_voice/
│   ├── frontend/                   # NAT front-end plugin (WebSocket server)
│   │   ├── config.py               # Pydantic config definition
│   │   ├── connection.py           # Per-connection handler (VAD→ASR→LLM→TTS)
│   │   ├── plugin.py               # NAT FrontEndBase implementation
│   │   ├── register.py             # NAT front-end registration
│   │   └── ws_server.py            # FastAPI WebSocket server
│   ├── pipeline/                   # Voice pipeline components
│   │   ├── asr.py                  # FunASR speech recognition
│   │   ├── tts.py                  # Edge TTS / CosyVoice synthesis
│   │   └── vad.py                  # Silero VAD voice activity detection
│   ├── tools/                      # Custom NAT tools
│   │   ├── register.py             # explain_scene registration (llm_name reference)
│   │   └── vlm_camera.py           # VLM camera tool
│   ├── utils/
│   │   ├── audio_codec.py          # Opus encode/decode
│   │   ├── audio_rate_controller.py # Audio rate control
│   │   └── auth.py                 # JWT authentication
│   └── workflow/
│       └── register.py             # LangGraph Agent definition (with memory compression)
├── py-xiaozhi-ws.py                # Desktop test client (push-to-talk)
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
