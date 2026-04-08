"""Pydantic config for the Xiaozhi Voice front end."""

from __future__ import annotations

from pydantic import Field

from nat.data_models.front_end import FrontEndBaseConfig


class XiaozhiVoiceFrontEndConfig(FrontEndBaseConfig, name="xiaozhi_voice"):
    """Config section: ``general.front_end._type: xiaozhi_voice``"""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    ws_path: str = Field(default="/xiaozhi/v1/")

    # Audio defaults sent in hello response
    audio_format: str = Field(default="opus")
    sample_rate: int = Field(default=24000)
    channels: int = Field(default=1)
    frame_duration: int = Field(default=60)

    # Auth
    auth_enabled: bool = Field(default=False)
    auth_secret_key: str = Field(default="nat-xiaozhi-secret")
    auth_expire_seconds: int = Field(default=2592000)  # 30 days
    auth_allowed_devices: list[str] = Field(default_factory=list)

    # Pipeline
    vad_model_dir: str = Field(default="models/silero_vad")
    vad_threshold: float = Field(default=0.5)
    vad_threshold_low: float = Field(default=0.2)
    vad_silence_ms: int = Field(default=1000)

    # ASR provider: "funasr" (local SenseVoice) or "qwen3_omni" (vLLM API)
    asr_provider: str = Field(default="funasr")
    asr_model_dir: str = Field(default="models/SenseVoiceSmall")
    # Qwen3-Omni ASR settings (used when asr_provider == "qwen3_omni")
    qwen3_omni_api_url: str = Field(default="http://localhost:8901/v1")
    qwen3_omni_model: str = Field(default="Qwen3-Omni-30B-A3B-Instruct")
    qwen3_omni_asr_prompt: str = Field(
        default="請將這段語音精確轉換為文字。只輸出辨識到的語音內容，不要加任何額外說明。"
    )
    qwen3_omni_use_data_url: bool = Field(default=False)

    tts_api_url: str = Field(default="http://127.0.0.1:50000/tts_stream")
    tts_spk_id: str = Field(default="default")

    # Pipeline mode: "separate" (ASR→LLM→TTS) or "e2e" (single Qwen3-Omni call)
    pipeline_mode: str = Field(default="separate")
    # When True and pipeline_mode=="e2e", uses streaming 3-stage API with
    # async_chunk for ~2s time-to-first-audio.  Requires vLLM-Omni started
    # with an async_chunk stage config (qwen3_omni_single_gpu_async.yaml).
    omni_e2e_streaming: bool = Field(default=False)
    omni_e2e_system_prompt: str = Field(
        default="You are Qwen, a virtual human capable of perceiving auditory inputs "
                "and generating text and speech. You are a helpful assistant."
    )
    omni_e2e_user_prompt: str = Field(default="請聽這段語音並回覆。")

    # E2E Tool Calling — enables Qwen3-Omni native function calling in E2E mode
    e2e_tool_calling: bool = Field(default=False)
    weather_api_host: str = Field(default="")
    weather_api_key: str = Field(default="")
    weather_default_location: str = Field(default="台北")

    # Timeouts
    close_no_voice_seconds: int = Field(default=120)

    # Workflow function name to invoke (must match a key in ``functions:`` section)
    workflow_function: str = Field(default="voice_agent")

    # CORS (for potential HTTP endpoints alongside WS)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
