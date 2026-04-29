"""USB webcam capture + VLM scene description.

Supports two camera sources:

* **Local camera** — uses ``cv2.VideoCapture`` on the NAT server.
* **Remote camera** — fetches a base64 JPEG from a remote Robot camera
  server via HTTP (e.g. ``camera_server.py`` running on the robot).

In both cases, the captured image is sent to a vision-language model
(VLM) running on the NAT server for scene description.

Compatible with reasoning models (e.g. Nemotron-3-Nano-Omni) — the
``<think>`` reasoning trace is stripped from the output automatically,
and ``enable_thinking`` is set to False for faster VLM responses.

Thread safety: all OpenCV calls are serialised through an asyncio Lock
and run in the default executor so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from typing import Optional

import cv2
import openai

logger = logging.getLogger(__name__)

_camera: Optional[cv2.VideoCapture] = None
_camera_index: int = 0
_camera_lock = asyncio.Lock()

_vlm_client: Optional[openai.AsyncOpenAI] = None
_vlm_model: str = "meta/llama-3.2-11b-vision-instruct"
_default_prompt: str = "用繁體中文列出畫面中所有物品和人。每項一行，包含位置(左/中/右/前/後)、顏色、大小或形狀。只列清單，不要寫開頭總結。"

_remote_camera_url: str = ""
_use_relay: bool = False

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def configure(
    *,
    camera_index: int = 0,
    vlm_api_key: str = "",
    vlm_base_url: str = "",
    vlm_model: str = "meta/llama-3.2-11b-vision-instruct",
    default_prompt: str = "用繁體中文列出畫面中所有物品和人。每項一行，包含位置(左/中/右/前/後)、顏色、大小或形狀。只列清單，不要寫開頭總結。",
    remote_camera_url: str = "",
    use_relay: bool = False,
) -> None:
    """Call once at startup to configure camera and VLM client.

    Parameters
    ----------
    remote_camera_url : str, optional
        URL of the remote Robot camera server (e.g. ``http://192.168.1.100:9903``).
        When set, the tool fetches images from the remote server instead of
        the local USB camera.  The ``/capture`` endpoint is appended automatically.
    use_relay : bool, optional
        When True, use the built-in WebSocket relay (robot_relay) instead of
        HTTP fetch.  Requires ``relay_enabled: true`` in front-end config.
    """
    global _camera_index, _vlm_client, _vlm_model, _default_prompt, _remote_camera_url, _use_relay

    _camera_index = camera_index
    _vlm_model = vlm_model
    _default_prompt = default_prompt
    _remote_camera_url = remote_camera_url.rstrip("/") if remote_camera_url else ""
    _use_relay = use_relay

    kwargs: dict = {"api_key": vlm_api_key}
    if vlm_base_url:
        kwargs["base_url"] = vlm_base_url
    _vlm_client = openai.AsyncOpenAI(**kwargs)

    if _use_relay:
        logger.info(
            "VLM camera configured: RELAY (built-in WebSocket) model=%s base_url=%s",
            vlm_model, vlm_base_url or "(default)",
        )
    elif _remote_camera_url:
        logger.info(
            "VLM camera configured: REMOTE=%s model=%s base_url=%s",
            _remote_camera_url, vlm_model, vlm_base_url or "(default)",
        )
    else:
        logger.info(
            "VLM camera configured: LOCAL camera=%d model=%s base_url=%s",
            camera_index, vlm_model, vlm_base_url or "(default)",
        )
        logger.info("Camera will be opened lazily on first tool call")


def _ensure_camera() -> cv2.VideoCapture:
    """Lazy-init the camera capture (called inside executor)."""
    global _camera
    if _camera is None or not _camera.isOpened():
        logger.info("Opening camera index=%d ...", _camera_index)
        _camera = cv2.VideoCapture(_camera_index)
        if not _camera.isOpened():
            raise RuntimeError(f"Cannot open USB webcam (index={_camera_index})")
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        import time as _t
        _t.sleep(0.3)
        logger.info("Camera opened successfully")
    return _camera


def _capture_b64() -> str:
    """Capture one frame and return JPEG base64 (blocking, run in executor)."""
    cap = _ensure_camera()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture image from USB webcam")
    frame = cv2.resize(frame, (480, 360))
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return base64.b64encode(buf).decode("utf-8")


async def _fetch_remote_image() -> str:
    """Fetch a base64 JPEG image from the remote Robot camera server."""
    import aiohttp

    url = f"{_remote_camera_url}/capture"
    logger.info("Fetching remote camera image from %s", url)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Remote camera returned HTTP {resp.status}")
                data = await resp.json()

        if not data.get("success"):
            error_msg = data.get("error", "Unknown remote camera error")
            raise RuntimeError(f"Remote camera error: {error_msg}")

        b64 = data["image_b64"]
        logger.info("Remote camera returned %d bytes base64", len(b64))
        return b64

    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Remote camera server ({_remote_camera_url}) timed out — "
            "camera may not be connected on the robot"
        )
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"Cannot reach remote camera server ({_remote_camera_url}): {exc}") from exc


async def _fetch_relay_image() -> str:
    """Fetch a base64 JPEG image via the built-in WebSocket relay."""
    from nat_xiaozhi_voice.frontend.ws_server import robot_relay

    logger.info("Fetching camera image via built-in relay")
    result = await robot_relay.capture(camera_index=0)

    if not result.get("success"):
        error_msg = result.get("error", "Unknown relay error")
        raise RuntimeError(f"Relay camera error: {error_msg}")

    b64 = result.get("image_b64", "")
    logger.info("Relay camera returned %d bytes base64", len(b64))
    return b64


async def _get_image_b64() -> str:
    """Get a base64 image from local camera, remote HTTP, or built-in relay."""
    if _use_relay:
        return await _fetch_relay_image()
    elif _remote_camera_url:
        return await _fetch_remote_image()
    else:
        loop = asyncio.get_running_loop()
        async with _camera_lock:
            return await loop.run_in_executor(None, _capture_b64)


async def analyze_scene(query: str = "") -> str:
    """Capture a frame and ask the VLM to describe it.

    Automatically chooses local or remote camera based on configuration.
    VLM analysis always runs on the NAT server side.

    Parameters
    ----------
    query : str, optional
        Extra instruction for the VLM (e.g. "look for red objects").
        If empty, ``_default_prompt`` is used.

    Returns
    -------
    str
        The VLM's text description.
    """
    if _vlm_client is None:
        return "攝影機工具尚未設定，無法使用。"

    t0 = time.time()

    try:
        b64_image = await _get_image_b64()
    except RuntimeError as exc:
        logger.warning("Camera capture failed: %s", exc)
        if _remote_camera_url:
            return f"無法連線遠端攝影機（{_remote_camera_url}）：{exc}"
        return f"無法開啟攝影機：{exc}。目前沒有可用的攝影機裝置。"

    t_cap = time.time()
    source = "relay" if _use_relay else ("remote" if _remote_camera_url else "local")
    logger.info("Camera capture (%s) took %.2fs", source, t_cap - t0)

    prompt_text = query.strip() if query.strip() else _default_prompt

    try:
        response = await asyncio.wait_for(
            _vlm_client.chat.completions.create(
                model=_vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                max_tokens=200,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - t_cap
        logger.warning("VLM API timed out after %.1fs", elapsed)
        return "攝影機有拍到畫面，但視覺分析超時了，請稍後再試。"
    except Exception as exc:
        logger.warning("VLM API call failed: %s", exc)
        return f"攝影機已擷取畫面，但 VLM 分析失敗：{exc}"

    msg = response.choices[0].message
    result = msg.content or ""
    result = _THINK_RE.sub("", result).strip()
    if not result:
        logger.warning(
            "VLM returned empty content. finish_reason=%s, refusal=%s, role=%s, raw=%s",
            response.choices[0].finish_reason,
            getattr(msg, "refusal", None),
            msg.role,
            str(msg)[:500],
        )
    logger.info("VLM analysis (%s camera) took %.2fs, result=%d chars", source, time.time() - t_cap, len(result))
    return result


def release_camera() -> None:
    """Release the camera resource (call at shutdown)."""
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None
        logger.info("Camera released")
