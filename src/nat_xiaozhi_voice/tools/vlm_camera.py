"""USB webcam capture + VLM scene description.

Manages a persistent cv2.VideoCapture and provides an async-safe
``analyze_scene`` function that captures a frame, encodes it to JPEG
base64, and sends it to a vision-language model for description.

Thread safety: all OpenCV calls are serialised through an asyncio Lock
and run in the default executor so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import base64
import logging
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
_default_prompt: str = "用繁體中文描述畫面，20字以內"


def configure(
    *,
    camera_index: int = 0,
    vlm_api_key: str = "",
    vlm_base_url: str = "",
    vlm_model: str = "meta/llama-3.2-11b-vision-instruct",
    default_prompt: str = "用繁體中文描述畫面，20字以內",
) -> None:
    """Call once at startup to configure camera and VLM client."""
    global _camera_index, _vlm_client, _vlm_model, _default_prompt

    _camera_index = camera_index
    _vlm_model = vlm_model
    _default_prompt = default_prompt

    kwargs: dict = {"api_key": vlm_api_key}
    if vlm_base_url:
        kwargs["base_url"] = vlm_base_url
    _vlm_client = openai.AsyncOpenAI(**kwargs)

    logger.info(
        "VLM camera configured: camera=%d model=%s base_url=%s",
        camera_index, vlm_model, vlm_base_url or "(default)",
    )

    # Skip pre-warm; camera is opened lazily on first tool call.
    # This avoids startup errors when no camera is attached.
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


async def analyze_scene(query: str = "") -> str:
    """Capture a frame and ask the VLM to describe it.

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
        loop = asyncio.get_running_loop()
        async with _camera_lock:
            b64_image = await loop.run_in_executor(None, _capture_b64)
    except RuntimeError as exc:
        logger.warning("Camera capture failed: %s", exc)
        return f"無法開啟攝影機：{exc}。目前沒有可用的攝影機裝置。"

    t_cap = time.time()
    logger.info("Camera capture took %.2fs", t_cap - t0)

    prompt_text = query.strip() if query.strip() else _default_prompt

    try:
        response = await _vlm_client.chat.completions.create(
            model=_vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    ],
                }
            ],
            max_tokens=256,
        )
    except Exception as exc:
        logger.warning("VLM API call failed: %s", exc)
        return f"攝影機已擷取畫面，但 VLM 分析失敗：{exc}"

    result = response.choices[0].message.content or ""
    logger.info("VLM analysis took %.2fs, result=%d chars", time.time() - t_cap, len(result))
    return result


def release_camera() -> None:
    """Release the camera resource (call at shutdown)."""
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None
        logger.info("Camera released")
