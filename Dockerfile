FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        build-essential \
        libopus-dev libopus0 libsndfile1-dev \
        git curl ca-certificates tzdata \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install uv for fast dependency resolution ────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ── Python dependencies (cached layer — only rebuilds when pyproject changes) ─
COPY pyproject.toml ./
RUN mkdir -p src/nat_xiaozhi_voice && \
    touch src/nat_xiaozhi_voice/__init__.py && \
    uv pip install --system -e . && \
    rm -rf src/nat_xiaozhi_voice

# ── Override PyTorch → CUDA 12.8 build ───────────────────────────────────────
RUN uv pip install --system \
        torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 --reinstall

# ── Download models ──────────────────────────────────────────────────────────
RUN mkdir -p models && \
    git clone --depth 1 https://github.com/snakers4/silero-vad.git models/snakers4_silero-vad && \
    python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('FunAudioLLM/SenseVoiceSmall', local_dir='models/SenseVoiceSmall')"

# ── Copy application source ─────────────────────────────────────────────────
COPY src/ src/
COPY configs/ configs/
COPY py-xiaozhi-ws.py test_vlm.py ./

# ── Re-install in editable mode with actual source ───────────────────────────
RUN uv pip install --system -e .

# ── NAT timezone config ─────────────────────────────────────────────────────
RUN mkdir -p /root/.config/nat && \
    echo '{"fallback_timezone": "system"}' > /root/.config/nat/config.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["nat", "start", "xiaozhi_voice"]
CMD ["--config_file", "/app/configs/xiaozhi_voice.yml"]
