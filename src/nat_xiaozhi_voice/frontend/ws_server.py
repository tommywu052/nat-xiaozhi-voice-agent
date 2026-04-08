"""Xiaozhi-compatible WebSocket server running inside NAT's front-end lifecycle.

Hosts a FastAPI application with:
- ``/xiaozhi/v1/`` — binary WebSocket for ESP32 / py-xiaozhi clients
- ``/health``      — simple health-check
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocket, WebSocketDisconnect

from nat_xiaozhi_voice.frontend.config import XiaozhiVoiceFrontEndConfig
from nat_xiaozhi_voice.frontend.connection import ConnectionHandler
from nat_xiaozhi_voice.pipeline.tts import CosyVoiceTTS
from nat_xiaozhi_voice.pipeline.vad import SileroVAD
from nat_xiaozhi_voice.utils.auth import AuthManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class XiaozhiWSServer:
    """Manages the FastAPI app, pipeline singletons, and active connections."""

    def __init__(
        self,
        config: XiaozhiVoiceFrontEndConfig,
        agent_fn: Callable[[str, str], Awaitable[str]],
    ):
        self._cfg = config
        self._agent_fn = agent_fn
        self._connections: dict[str, ConnectionHandler] = {}

        # Auth
        self._auth: AuthManager | None = None
        if config.auth_enabled:
            self._auth = AuthManager(config.auth_secret_key, config.auth_expire_seconds)

        # Pipeline singletons (lazy-init in startup)
        self._vad: SileroVAD | None = None
        self._asr: Any = None  # FunASRRecognizer | Qwen3OmniASR
        self._tts: CosyVoiceTTS | None = None
        self._omni_e2e: Any = None  # Qwen3OmniE2E (when pipeline_mode == "e2e")
        self._tool_executor: Any = None  # E2EToolExecutor (when e2e_tool_calling)

        # Welcome template
        self._welcome = {
            "type": "hello",
            "version": 1,
            "transport": "websocket",
            "audio_params": {
                "format": config.audio_format,
                "sample_rate": config.sample_rate,
                "channels": config.channels,
                "frame_duration": config.frame_duration,
            },
        }

        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Xiaozhi Voice Agent")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self._cfg.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.add_event_handler("startup", self._startup)
        app.add_event_handler("shutdown", self._shutdown)
        app.add_api_route("/health", self._health, methods=["GET"])

        ws_path = self._cfg.ws_path.rstrip("/") + "/"
        app.add_api_websocket_route(ws_path, self._ws_endpoint)
        ws_path_no_slash = ws_path.rstrip("/")
        if ws_path_no_slash != ws_path:
            app.add_api_websocket_route(ws_path_no_slash, self._ws_endpoint)

        return app

    # ── lifecycle ──────────────────────────────────────────────────────

    async def _startup(self):
        logger.info("Initializing voice pipeline …")
        self._vad = SileroVAD(
            self._cfg.vad_model_dir,
            threshold=self._cfg.vad_threshold,
            threshold_low=self._cfg.vad_threshold_low,
            silence_ms=self._cfg.vad_silence_ms,
        )

        pipeline_mode = self._cfg.pipeline_mode.lower()
        if pipeline_mode == "e2e":
            from nat_xiaozhi_voice.pipeline.omni_e2e import Qwen3OmniE2E

            system_prompt = self._cfg.omni_e2e_system_prompt
            if self._cfg.e2e_tool_calling:
                from nat_xiaozhi_voice.pipeline.tools import (
                    E2EToolExecutor,
                    build_tool_prompt_section,
                )
                system_prompt = system_prompt.rstrip() + build_tool_prompt_section()
                self._tool_executor = E2EToolExecutor(
                    weather_api_host=self._cfg.weather_api_host,
                    weather_api_key=self._cfg.weather_api_key,
                    weather_default_location=self._cfg.weather_default_location,
                )
                from nat_xiaozhi_voice.pipeline.tools import TOOL_DEFS
                logger.info("E2E tool calling enabled: %d tools (%s)",
                            len(TOOL_DEFS),
                            ", ".join(t["function"]["name"] for t in TOOL_DEFS))

            self._omni_e2e = Qwen3OmniE2E(
                api_url=self._cfg.qwen3_omni_api_url,
                model=self._cfg.qwen3_omni_model,
                system_prompt=system_prompt,
                user_audio_prompt=self._cfg.omni_e2e_user_prompt,
                base_system_prompt=self._cfg.omni_e2e_system_prompt,
            )
            mode_desc = "VAD + E2E Qwen3-Omni"
            if self._cfg.omni_e2e_streaming:
                mode_desc += " async_chunk streaming"
            if self._cfg.e2e_tool_calling:
                mode_desc += " + tool calling"
            logger.info("Voice pipeline ready (%s)", mode_desc)
        else:
            asr_provider = self._cfg.asr_provider.lower()
            if asr_provider == "qwen3_omni":
                from nat_xiaozhi_voice.pipeline.asr_qwen3_omni import Qwen3OmniASR

                self._asr = Qwen3OmniASR(
                    api_url=self._cfg.qwen3_omni_api_url,
                    model=self._cfg.qwen3_omni_model,
                    asr_prompt=self._cfg.qwen3_omni_asr_prompt,
                    use_data_url=self._cfg.qwen3_omni_use_data_url,
                )
            else:
                from nat_xiaozhi_voice.pipeline.asr import FunASRRecognizer

                self._asr = FunASRRecognizer(self._cfg.asr_model_dir)

            self._tts = CosyVoiceTTS(self._cfg.tts_api_url, self._cfg.tts_spk_id)
            logger.info("Voice pipeline ready (VAD + ASR[%s] + TTS)", asr_provider)

    async def _shutdown(self):
        logger.info("Shutting down voice server, %d active connections", len(self._connections))
        self._connections.clear()

    # ── routes ─────────────────────────────────────────────────────────

    async def _health(self):
        mode = "separate"
        if self._omni_e2e:
            mode = "e2e_streaming" if self._cfg.omni_e2e_streaming else "e2e"
        return {
            "status": "ok",
            "connections": len(self._connections),
            "pipeline": {
                "mode": mode,
                "vad": self._vad is not None,
                "asr": self._asr is not None,
                "tts": self._tts is not None,
                "omni_e2e": self._omni_e2e is not None,
                "streaming": self._cfg.omni_e2e_streaming,
            },
        }

    async def _ws_endpoint(self, ws: WebSocket):
        # Extract device headers (from query params or headers)
        device_id = (
            ws.query_params.get("device-id")
            or ws.headers.get("device-id", "")
            or ws.headers.get("Device-Id", "")
            or "unknown"
        )
        client_id = (
            ws.query_params.get("client-id")
            or ws.headers.get("client-id", "")
            or ws.headers.get("Client-Id", "")
            or ""
        )

        # Auth check
        if self._auth and self._cfg.auth_enabled:
            if device_id not in self._cfg.auth_allowed_devices:
                auth_header = ws.headers.get("authorization", "")
                token = auth_header.replace("Bearer ", "").strip() if auth_header else ""
                if not token or not self._auth.verify_token(token, client_id, device_id):
                    await ws.close(code=4003, reason="Unauthorized")
                    logger.warning("Auth failed: device=%s client=%s", device_id, client_id)
                    return

        await ws.accept()
        logger.info("Client connected: device=%s client=%s", device_id, client_id)

        handler = ConnectionHandler(
            ws,
            vad=self._vad,  # type: ignore[arg-type]
            asr=self._asr,  # type: ignore[arg-type]
            tts=self._tts,  # type: ignore[arg-type]
            agent_fn=self._agent_fn,
            welcome_msg=self._welcome,
            close_no_voice_seconds=self._cfg.close_no_voice_seconds,
            omni_e2e=self._omni_e2e,
            omni_e2e_streaming=self._cfg.omni_e2e_streaming,
            tool_executor=self._tool_executor,
        )
        handler.device_id = device_id
        handler.client_id = client_id
        self._connections[handler.session_id] = handler

        try:
            await handler.run()
        finally:
            self._connections.pop(handler.session_id, None)
