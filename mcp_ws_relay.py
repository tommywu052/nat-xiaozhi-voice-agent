#!/usr/bin/env python3
"""MCP WebSocket Relay — HTTP-to-WebSocket bridge for remote camera.

Solves the dynamic IP problem: Robot actively connects INTO the relay via
WebSocket, so the NAT server never needs to know the Robot's IP.

Architecture::

    Robot (dynamic IP)              NAT Server (this relay)
    ┌──────────────┐               ┌───────────────────────┐
    │ camera_server │──WS connect─→│ mcp_ws_relay.py       │
    │ --relay ws:// │              │  WS  /ws/robot        │
    └──────────────┘               │  HTTP /capture        │←── NAT vlm_camera
                                   │  HTTP /health         │
                                   └───────────────────────┘

Usage::

    # On NAT server (start relay):
    python mcp_ws_relay.py --port 9903

    # On Robot (connect to relay):
    python camera_server.py --relay ws://<NAT_SERVER_IP>:9903/ws/robot

    # NAT config (xiaozhi_voice.yml):
    remote_camera_url: "http://localhost:9903"

Dependencies::

    pip install fastapi uvicorn websockets
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("MCPWSRelay")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="MCP WebSocket Relay")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Robot connection state ──────────────────────────────────────────────

_robot_ws: Optional[WebSocket] = None
_robot_lock = asyncio.Lock()
_pending_requests: dict[str, asyncio.Future] = {}


async def _send_to_robot(request_id: str, payload: dict, timeout: float = 15.0) -> dict:
    """Send a request to the Robot via WebSocket and wait for the response."""
    if _robot_ws is None:
        raise RuntimeError("No robot connected")

    future: asyncio.Future = asyncio.get_event_loop().create_future()
    _pending_requests[request_id] = future

    try:
        await _robot_ws.send_json({"request_id": request_id, **payload})
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError("Robot did not respond within timeout")
    finally:
        _pending_requests.pop(request_id, None)


# ── WebSocket endpoint (Robot connects here) ────────────────────────────

@app.websocket("/ws/robot")
async def robot_websocket(ws: WebSocket):
    """Accept a WebSocket connection from the Robot."""
    global _robot_ws

    await ws.accept()
    remote = ws.client
    logger.info("Robot connected from %s:%s", remote.host if remote else "?", remote.port if remote else "?")

    async with _robot_lock:
        if _robot_ws is not None:
            logger.warning("Replacing existing robot connection")
            try:
                await _robot_ws.close(code=1000, reason="replaced by new connection")
            except Exception:
                pass
        _robot_ws = ws

    try:
        while True:
            data = await ws.receive_json()

            request_id = data.get("request_id")
            if request_id and request_id in _pending_requests:
                _pending_requests[request_id].set_result(data)
            else:
                logger.warning("Received unexpected message: %s", str(data)[:200])

    except WebSocketDisconnect:
        logger.info("Robot disconnected")
    except Exception as e:
        logger.error("Robot WebSocket error: %s", e)
    finally:
        async with _robot_lock:
            if _robot_ws is ws:
                _robot_ws = None
        for rid, fut in list(_pending_requests.items()):
            if not fut.done():
                fut.set_exception(RuntimeError("Robot disconnected"))
            _pending_requests.pop(rid, None)


# ── HTTP endpoints (NAT calls these) ────────────────────────────────────

@app.get("/capture")
async def capture(index: int = Query(0, description="Camera device index")):
    """Forward a capture request to the connected Robot."""
    if _robot_ws is None:
        return {"success": False, "error": "No robot connected — waiting for robot to connect via WebSocket"}

    request_id = str(uuid.uuid4())
    try:
        result = await _send_to_robot(
            request_id,
            {"type": "capture", "camera_index": index},
        )
        return {
            "success": result.get("success", False),
            "image_b64": result.get("image_b64", ""),
            "error": result.get("error", ""),
        }
    except RuntimeError as e:
        logger.error("Capture relay failed: %s", e)
        return {"success": False, "error": str(e)}


@app.get("/health")
async def health():
    """Health check — reports robot connection status."""
    return {
        "status": "ok",
        "robot_connected": _robot_ws is not None,
    }


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="MCP WebSocket Relay")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=9903, help="Bind port")
    args = parser.parse_args()

    logger.info("Starting MCP WebSocket Relay on %s:%d", args.host, args.port)
    logger.info("  Robot connects to:  ws://<this-host>:%d/ws/robot", args.port)
    logger.info("  NAT calls:          http://localhost:%d/capture", args.port)
    logger.info("  Health check:       http://localhost:%d/health", args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
