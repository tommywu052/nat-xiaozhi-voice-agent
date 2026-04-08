"""Xiaozhi-compatible WebSocket server running inside NAT's front-end lifecycle.

Hosts a FastAPI application with:
- ``/xiaozhi/v1/``                — binary WebSocket for ESP32 / py-xiaozhi clients
- ``/health``                     — simple health-check
- ``/api/memory/{device_id}``     — DELETE to clear per-device memory
- ``/api/memory``                 — DELETE to clear all memory
- ``/api/memory``                 — GET to list all devices with memory
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocket, WebSocketDisconnect

from nat_xiaozhi_voice.frontend.config import XiaozhiVoiceFrontEndConfig
from nat_xiaozhi_voice.frontend.connection import ConnectionHandler
from nat_xiaozhi_voice.pipeline.asr import FunASRRecognizer
from nat_xiaozhi_voice.pipeline.tts import CosyVoiceTTS, EdgeTTS
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
        agent_stream_fn=None,
        clear_memory_fn: Optional[Callable[[str], Awaitable[None]]] = None,
        clear_all_memory_fn: Optional[Callable[[], Awaitable[None]]] = None,
        list_memory_devices_fn: Optional[Callable[[], Awaitable[list[str]]]] = None,
    ):
        self._cfg = config
        self._agent_fn = agent_fn
        self._agent_stream_fn = agent_stream_fn
        self._clear_memory_fn = clear_memory_fn
        self._clear_all_memory_fn = clear_all_memory_fn
        self._list_memory_devices_fn = list_memory_devices_fn
        self._connections: dict[str, ConnectionHandler] = {}

        # Auth
        self._auth: AuthManager | None = None
        if config.auth_enabled:
            self._auth = AuthManager(config.auth_secret_key, config.auth_expire_seconds)

        # Pipeline singletons (lazy-init in startup)
        self._vad: SileroVAD | None = None
        self._asr: FunASRRecognizer | None = None
        self._tts: CosyVoiceTTS | None = None

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
        app.add_api_route("/api/memory", self._list_memory, methods=["GET"])
        app.add_api_route("/api/memory", self._clear_all_memory, methods=["DELETE"])
        app.add_api_route("/api/memory/{device_id:path}", self._clear_device_memory, methods=["DELETE"])

        ws_path = self._cfg.ws_path.rstrip("/") + "/"
        app.add_api_websocket_route(ws_path, self._ws_endpoint)
        ws_path_no_slash = ws_path.rstrip("/")
        if ws_path_no_slash != ws_path:
            app.add_api_websocket_route(ws_path_no_slash, self._ws_endpoint)

        return app

    # ── lifecycle ──────────────────────────────────────────────────────

    async def _startup(self):
        logger.info("Initializing voice pipeline ...")
        self._vad = SileroVAD(
            self._cfg.vad_model_dir,
            threshold=self._cfg.vad_threshold,
            threshold_low=self._cfg.vad_threshold_low,
            silence_ms=self._cfg.vad_silence_ms,
        )
        self._asr = FunASRRecognizer(self._cfg.asr_model_dir)
        if self._cfg.tts_type == "edge":
            self._tts = EdgeTTS(self._cfg.tts_voice)
        else:
            self._tts = CosyVoiceTTS(self._cfg.tts_api_url, self._cfg.tts_spk_id)
        logger.info("Voice pipeline ready (VAD + ASR + TTS)")

        # Warm up LLM connection in background (non-blocking)
        asyncio.create_task(self._warmup_llm())

    async def _shutdown(self):
        logger.info("Shutting down voice server, %d active connections", len(self._connections))
        self._connections.clear()

    async def _warmup_llm(self):
        """Send a dummy agent call to warm up the LLM connection pool."""
        if not self._agent_fn:
            return
        try:
            import time
            t0 = time.time()
            logger.info("Warming up LLM connection...")
            await self._agent_fn("ping", "__warmup__")
            logger.info("LLM warm-up done in %.2fs", time.time() - t0)
        except Exception as e:
            logger.warning("LLM warm-up failed (non-critical): %s", e)

    # ── routes ─────────────────────────────────────────────────────────

    async def _health(self):
        return {
            "status": "ok",
            "connections": len(self._connections),
            "pipeline": {
                "vad": self._vad is not None,
                "asr": self._asr is not None,
                "tts": self._tts is not None,
            },
        }

    async def _list_memory(self):
        """GET /api/memory — list devices that have stored memory."""
        if not self._list_memory_devices_fn:
            return {"error": "Memory persistence not enabled"}
        try:
            devices = await self._list_memory_devices_fn()
            return {"devices": devices, "count": len(devices)}
        except Exception:
            logger.exception("Failed to list memory devices")
            return {"error": "Internal error"}

    async def _clear_device_memory(self, device_id: str):
        """DELETE /api/memory/{device_id} — clear memory for one device."""
        if not self._clear_memory_fn:
            return {"error": "Memory persistence not enabled"}
        try:
            await self._clear_memory_fn(device_id)
            logger.info("REST API: cleared memory for device=%s", device_id)
            return {"status": "ok", "device_id": device_id, "action": "cleared"}
        except Exception:
            logger.exception("Failed to clear memory for device=%s", device_id)
            return {"error": "Internal error"}

    async def _clear_all_memory(self):
        """DELETE /api/memory — clear all device memory."""
        if not self._clear_all_memory_fn:
            return {"error": "Memory persistence not enabled"}
        try:
            await self._clear_all_memory_fn()
            logger.info("REST API: cleared ALL memory")
            return {"status": "ok", "action": "all_cleared"}
        except Exception:
            logger.exception("Failed to clear all memory")
            return {"error": "Internal error"}

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
            agent_stream_fn=self._agent_stream_fn,
            clear_memory_fn=self._clear_memory_fn,
            welcome_msg=self._welcome,
            close_no_voice_seconds=self._cfg.close_no_voice_seconds,
        )
        handler.device_id = device_id
        handler.client_id = client_id
        self._connections[handler.session_id] = handler

        try:
            await handler.run()
        finally:
            self._connections.pop(handler.session_id, None)
