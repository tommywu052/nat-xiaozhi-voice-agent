"""Xiaozhi-compatible WebSocket server running inside NAT's front-end lifecycle.

Hosts a FastAPI application with:
- ``/xiaozhi/v1/``                — binary WebSocket for ESP32 / py-xiaozhi clients
- ``/ws/robot``                   — WebSocket relay for Robot camera (when relay_enabled)
- ``/health``                     — simple health-check
- ``/api/memory/{device_id}``     — DELETE to clear per-device memory
- ``/api/memory``                 — DELETE to clear all memory
- ``/api/memory``                 — GET to list all devices with memory
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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


# ── Robot Camera Relay (singleton, shared across the process) ────────────

class RobotCameraRelay:
    """WebSocket relay bridge: Robot connects in, NAT captures via internal call.

    This lives as a module-level singleton so that both the FastAPI routes
    (in ws_server.py) and vlm_camera.py can access it without circular imports.
    """

    def __init__(self):
        self._robot_ws: Optional[WebSocket] = None
        self._lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future] = {}

    @property
    def robot_connected(self) -> bool:
        return self._robot_ws is not None

    async def set_robot(self, ws: Optional[WebSocket]):
        async with self._lock:
            if self._robot_ws is not None and ws is not self._robot_ws:
                try:
                    await self._robot_ws.close(1000, "replaced")
                except Exception:
                    pass
            self._robot_ws = ws

    async def remove_robot(self, ws: WebSocket):
        async with self._lock:
            if self._robot_ws is ws:
                self._robot_ws = None
        for rid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(RuntimeError("Robot disconnected"))
            self._pending.pop(rid, None)

    def resolve(self, request_id: str, data: dict):
        fut = self._pending.get(request_id)
        if fut and not fut.done():
            fut.set_result(data)

    async def capture(self, camera_index: int = 0, timeout: float = 15.0) -> dict:
        """Send a capture request to the Robot and wait for the response."""
        if self._robot_ws is None:
            return {"success": False, "error": "No robot connected — waiting for robot to connect via WebSocket"}

        request_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        try:
            await self._robot_ws.send_json({
                "request_id": request_id,
                "type": "capture",
                "camera_index": camera_index,
            })
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return {"success": False, "error": "Robot did not respond within timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self._pending.pop(request_id, None)


# Module-level singleton — importable by vlm_camera.py
robot_relay = RobotCameraRelay()


class SpeakRequest(BaseModel):
    text: str
    device_id: str = ""


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

        app.add_api_route("/api/speak", self._speak, methods=["POST"])

        # Proxy-friendly routes (no /api/ prefix) for NAT UI access
        app.add_api_route("/memory", self._list_memory, methods=["GET"])
        app.add_api_route("/memory", self._clear_all_memory, methods=["DELETE"])
        app.add_api_route("/memory/{device_id:path}", self._clear_device_memory, methods=["DELETE"])
        app.add_api_route("/history/{device_id:path}", self._get_history, methods=["GET"])
        app.add_api_route("/connections", self._get_connections, methods=["GET"])

        # Robot camera relay (when enabled)
        if self._cfg.relay_enabled:
            app.add_api_websocket_route("/ws/robot", self._robot_ws_endpoint)
            app.add_api_route("/relay/capture", self._relay_capture, methods=["GET"])
            logger.info("Robot camera relay enabled — Robot connects to ws://host:%d/ws/robot", self._cfg.port)

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
        result = {
            "status": "ok",
            "connections": len(self._connections),
            "pipeline": {
                "vad": self._vad is not None,
                "asr": self._asr is not None,
                "tts": self._tts is not None,
            },
        }
        if self._cfg.relay_enabled:
            result["robot_connected"] = robot_relay.robot_connected
        return result

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

    async def _get_history(self, device_id: str):
        """GET /history/{device_id} — retrieve conversation history from LangGraph."""
        from nat_xiaozhi_voice.workflow.register import _shared_agent_state

        agent = _shared_agent_state.get("agent")
        if not agent:
            return {"device_id": device_id, "messages": [], "count": 0, "error": "Agent not initialized"}

        try:
            from langchain_core.messages import AIMessage, HumanMessage

            state = await agent.aget_state({"configurable": {"thread_id": device_id}})
            if not state or not state.values:
                return {"device_id": device_id, "messages": [], "count": 0}

            messages = state.values.get("messages", [])
            result = []
            for m in messages:
                content = getattr(m, "content", "")
                if isinstance(content, list):
                    parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                    content = "".join(parts)
                else:
                    content = str(content)

                if not content.strip():
                    continue

                if isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    if getattr(m, "tool_calls", None):
                        continue
                    role = "assistant"
                else:
                    continue

                result.append({"role": role, "content": content})

            return {"device_id": device_id, "messages": result, "count": len(result)}
        except Exception:
            logger.exception("Failed to get history for device=%s", device_id)
            return {"device_id": device_id, "messages": [], "count": 0}

    async def _get_connections(self):
        """GET /connections — list currently active WebSocket connections."""
        return {
            "connections": [
                {
                    "device_id": h.device_id,
                    "client_id": h.client_id,
                    "session_id": h.session_id,
                }
                for h in self._connections.values()
            ],
            "count": len(self._connections),
        }

    async def _speak(self, req: SpeakRequest):
        """POST /api/speak — proactively push TTS to connected client(s)."""
        if not req.text.strip():
            return {"status": "error", "message": "text is empty"}

        targets = []
        if req.device_id:
            for h in self._connections.values():
                if h.device_id == req.device_id:
                    targets.append(h)
        else:
            targets = list(self._connections.values())

        if not targets:
            logger.warning("/api/speak: no connected clients (device_id=%s)", req.device_id or "*")
            return {"status": "error", "message": "no connected clients"}

        for h in targets:
            asyncio.create_task(h.speak(req.text))

        logger.info("/api/speak: pushing to %d client(s): %s", len(targets), req.text[:60])
        return {"status": "ok", "targets": len(targets)}

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

    # ── Robot camera relay endpoints ──────────────────────────────────

    async def _robot_ws_endpoint(self, ws: WebSocket):
        """Accept a WebSocket connection from the Robot camera."""
        await ws.accept()
        remote = ws.client
        logger.info("Robot camera connected from %s:%s",
                     remote.host if remote else "?", remote.port if remote else "?")

        await robot_relay.set_robot(ws)
        try:
            while True:
                data = await ws.receive_json()
                request_id = data.get("request_id", "")
                if request_id:
                    robot_relay.resolve(request_id, data)
                else:
                    logger.warning("Robot sent message without request_id: %s", str(data)[:200])
        except WebSocketDisconnect:
            logger.info("Robot camera disconnected")
        except Exception as e:
            logger.error("Robot camera WebSocket error: %s", e)
        finally:
            await robot_relay.remove_robot(ws)

    async def _relay_capture(self, index: int = Query(0, description="Camera device index")):
        """HTTP endpoint for internal relay capture (used by vlm_camera)."""
        return await robot_relay.capture(camera_index=index)
