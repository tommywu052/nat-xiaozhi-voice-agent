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
    asr_model_dir: str = Field(default="models/SenseVoiceSmall")
    tts_type: str = Field(default="cosyvoice", description="TTS backend: cosyvoice | edge")
    tts_api_url: str = Field(default="http://127.0.0.1:50000/tts_stream")
    tts_spk_id: str = Field(default="default")
    tts_voice: str = Field(default="zh-TW-HsiaoChenNeural", description="Edge TTS voice name")

    # Timeouts
    close_no_voice_seconds: int = Field(default=120)

    # Workflow function name to invoke (must match a key in ``functions:`` section)
    workflow_function: str = Field(default="voice_agent")

    # Robot camera relay — accept WebSocket from Robot on /ws/robot
    relay_enabled: bool = Field(
        default=False,
        description="Enable WebSocket relay for robot camera. Robot connects to ws://host:port/ws/robot",
    )

    # CORS (for potential HTTP endpoints alongside WS)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
