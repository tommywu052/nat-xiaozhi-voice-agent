#!/usr/bin/env python3
"""Robot Camera Server — captures images from USB camera.

Supports three modes:

1. **HTTP REST** (default, for NAT explain_scene tool — direct connection):

       python camera_server.py --port 9903

   Endpoints:
     GET  /capture            → {"success": true, "image_b64": "..."}
     GET  /capture?index=1    → use camera index 1
     GET  /health             → {"status": "ok"}

2. **MCP stdio** (for use with mcp_pipe.py + 小智 cloud):

       python camera_server.py --mcp

   Exposes ``capture_image`` tool via MCP stdio transport.

3. **WebSocket Relay client** (connects to NAT relay — no need for fixed Robot IP):

       python camera_server.py --relay ws://nat-server:9903/ws/robot

   Robot actively connects to the relay server; the relay bridges HTTP
   requests from NAT to this WebSocket connection.

Dependencies:
    pip install opencv-python fastapi uvicorn mcp websockets
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
import threading
from typing import Optional

import cv2

logger = logging.getLogger("CameraServer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Persistent camera manager — keeps the camera open to avoid slow re-init
# ---------------------------------------------------------------------------

_cam_lock = threading.Lock()
_cam: Optional[cv2.VideoCapture] = None
_cam_index: int = -1


def _get_camera(camera_index: int) -> cv2.VideoCapture:
    """Return a persistent VideoCapture, opening it on first use."""
    global _cam, _cam_index

    with _cam_lock:
        if _cam is not None and _cam.isOpened() and _cam_index == camera_index:
            return _cam

        if _cam is not None:
            _cam.release()
            _cam = None

        logger.info("Opening camera index=%d ...", camera_index)
        t0 = time.time()

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera (index={camera_index})")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm up: discard first few frames (auto-exposure settling)
        for _ in range(3):
            cap.read()

        _cam = cap
        _cam_index = camera_index
        logger.info("Camera opened in %.2fs", time.time() - t0)
        return _cam


def _release_camera():
    """Release the persistent camera."""
    global _cam, _cam_index
    with _cam_lock:
        if _cam is not None:
            _cam.release()
            _cam = None
            _cam_index = -1
            logger.info("Camera released")


# ---------------------------------------------------------------------------
# Core capture logic (shared by both modes)
# ---------------------------------------------------------------------------

def capture_image(
    camera_index: int = 0,
    quality: int = 75,
) -> dict:
    """Capture one frame from USB camera, return base64 JPEG."""
    t0 = time.time()
    try:
        cap = _get_camera(camera_index)

        with _cam_lock:
            ret, frame = cap.read()

        if not ret or frame is None:
            # Camera may have disconnected; release and retry once
            _release_camera()
            cap = _get_camera(camera_index)
            with _cam_lock:
                ret, frame = cap.read()
            if not ret or frame is None:
                return {"success": False, "error": "Failed to read frame from camera"}

        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        b64 = base64.b64encode(buf).decode("utf-8")

        elapsed = time.time() - t0
        logger.info(
            "Captured %dx%d image (quality=%d, %d bytes b64, %.2fs)",
            frame.shape[1], frame.shape[0], quality, len(b64), elapsed,
        )
        return {"success": True, "image_b64": b64}

    except Exception as e:
        logger.error("Capture error: %s", e)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Mode 1: HTTP REST server (for NAT explain_scene tool)
# ---------------------------------------------------------------------------

def run_http(host: str, port: int):
    """Start a FastAPI server with /capture endpoint."""
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    app = FastAPI(title="Robot Camera Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def startup():
        try:
            _get_camera(0)
            logger.info("Camera pre-warmed and ready")
        except Exception as e:
            logger.warning("Camera pre-warm failed (will retry on first capture): %s", e)

    @app.on_event("shutdown")
    def shutdown():
        _release_camera()

    @app.get("/capture")
    def api_capture(index: int = Query(0, description="Camera device index")):
        return capture_image(camera_index=index)

    @app.get("/health")
    def health():
        return {"status": "ok", "camera_open": _cam is not None and _cam.isOpened()}

    logger.info("Starting HTTP camera server on %s:%d", host, port)
    logger.info("  GET /capture          → capture image (base64)")
    logger.info("  GET /capture?index=1  → use camera index 1")
    logger.info("  GET /health           → health check")
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Mode 2: MCP stdio server (for mcp_pipe.py backward compatibility)
# ---------------------------------------------------------------------------

def run_mcp():
    """Start as MCP stdio server."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("RobotCamera")

    @mcp.tool()
    def mcp_capture_image(camera_index: int = 0) -> dict:
        """Capture a photo from the robot's USB camera and return the
        base64-encoded JPEG image. Only returns the raw image — VLM
        analysis should be performed by the caller."""
        return capture_image(camera_index=camera_index)

    logger.info("Starting MCP stdio camera server")
    mcp.run(transport="stdio")


# ---------------------------------------------------------------------------
# Mode 3: WebSocket Relay client (Robot → Relay, no fixed IP needed)
# ---------------------------------------------------------------------------

def run_relay(relay_url: str):
    """Connect to the MCP WebSocket Relay and respond to capture requests.

    The relay server runs on the NAT server side. This Robot connects
    to it via WebSocket, so the NAT server never needs to know the
    Robot's IP address.
    """
    import asyncio

    try:
        import websockets
    except ImportError:
        logger.error("websockets package required: pip install websockets")
        sys.exit(1)

    async def _relay_loop():
        reconnect_delay = 1.0
        max_reconnect_delay = 30.0

        while True:
            try:
                logger.info("Connecting to relay: %s", relay_url)
                async with websockets.connect(relay_url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("Connected to relay successfully")
                    reconnect_delay = 1.0

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON from relay: %s", raw[:200])
                            continue

                        request_id = msg.get("request_id", "")
                        msg_type = msg.get("type", "")

                        if msg_type == "capture":
                            camera_index = msg.get("camera_index", 0)
                            logger.info("Relay capture request (id=%s, cam=%d)", request_id, camera_index)
                            result = capture_image(camera_index=camera_index)
                            result["request_id"] = request_id
                            await ws.send(json.dumps(result))
                        else:
                            logger.warning("Unknown relay message type: %s", msg_type)
                            await ws.send(json.dumps({
                                "request_id": request_id,
                                "success": False,
                                "error": f"Unknown request type: {msg_type}",
                            }))

            except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
                logger.warning("Relay connection lost: %s. Reconnecting in %.0fs...", e, reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
            except Exception as e:
                logger.error("Unexpected relay error: %s. Reconnecting in %.0fs...", e, reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    # Pre-warm camera before entering relay loop
    try:
        _get_camera(0)
        logger.info("Camera pre-warmed for relay mode")
    except Exception as e:
        logger.warning("Camera pre-warm failed (will retry on capture): %s", e)

    asyncio.run(_relay_loop())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robot Camera Server (HTTP REST, MCP stdio, or WebSocket Relay)",
    )
    parser.add_argument(
        "--mcp", action="store_true",
        help="Run as MCP stdio server (for mcp_pipe.py)",
    )
    parser.add_argument(
        "--relay", type=str, default="",
        help="WebSocket relay URL (e.g. ws://nat-server:9903/ws/robot)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=9903, help="HTTP bind port")
    args = parser.parse_args()

    if args.mcp:
        run_mcp()
    elif args.relay:
        run_relay(args.relay)
    else:
        run_http(args.host, args.port)
