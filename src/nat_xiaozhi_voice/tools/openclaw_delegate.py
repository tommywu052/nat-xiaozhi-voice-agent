"""Delegate tasks to the local OpenClaw Agent via its CLI.

Auto-routes between sync and async based on keywords in the message,
so the LLM only needs to decide *whether* to call the tool — not *how*.

Async mode now waits for the result in a background task and pushes
it back to the voice client via POST /api/speak.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex

import aiohttp

logger = logging.getLogger(__name__)

_NVM_DIR = os.environ.get("NVM_DIR", os.path.expanduser("~/.nvm"))
_NODE_VERSION = os.environ.get("OPENCLAW_NODE_VERSION", "22")
_SESSION_ID = "voice-bridge"
_SYNC_TIMEOUT = 120
_ASYNC_TIMEOUT = 600
_SPEAK_URL = "http://127.0.0.1:8000/api/speak"
_last_user_text: str = ""

_SHELL_PREFIX = (
    f'export NVM_DIR="{_NVM_DIR}" '
    f'&& [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" '
    f"&& nvm use {_NODE_VERSION} --silent"
)

_NOISE_PREFIXES = (
    "Now using",
    "Registered plugin",
    "EMBEDDED FALLBACK",
)

_ASYNC_KEYWORDS = re.compile(
    r"(whatsapp|line|telegram|discord|傳送|傳到|發到|發送|送到|"
    r"排程|定時|每天|每小時|每\d+|cron|鬧鐘|提醒我|"
    r"報告|寫一份|整理成|生成文件|"
    r"研究|調查|分析|背景|深入|詳細|"
    r"research|investigate|analyze|background)",
    re.IGNORECASE,
)


def configure(
    *,
    session_id: str = "voice-bridge",
    sync_timeout: int | None = None,
    async_timeout: int | None = None,
) -> None:
    global _SESSION_ID, _SYNC_TIMEOUT, _ASYNC_TIMEOUT
    _SESSION_ID = session_id
    if sync_timeout is not None:
        _SYNC_TIMEOUT = sync_timeout
    if async_timeout is not None:
        _ASYNC_TIMEOUT = async_timeout
    logger.info(
        "openclaw_delegate configured: session=%s sync=%ds async=%ds",
        _SESSION_ID, _SYNC_TIMEOUT, _ASYNC_TIMEOUT,
    )


def set_last_user_text(text: str) -> None:
    """Called by the agent layer to track the latest user utterance."""
    global _last_user_text
    _last_user_text = text


def should_async(message: str) -> bool:
    return bool(_ASYNC_KEYWORDS.search(message))


def _build_cmd(message: str, timeout: int, *, fresh_session: bool = False) -> str:
    safe_msg = shlex.quote(message)
    if fresh_session:
        import uuid
        sid = f"{_SESSION_ID}-{uuid.uuid4().hex[:8]}"
    else:
        sid = _SESSION_ID
    return (
        f"{_SHELL_PREFIX} "
        f"&& openclaw agent --session-id {shlex.quote(sid)} "
        f"--message {safe_msg} --timeout {timeout}"
    )


def _clean_output(raw: str) -> str:
    lines = raw.strip().splitlines()
    cleaned = [l for l in lines if not any(l.startswith(p) for p in _NOISE_PREFIXES)]
    return "\n".join(cleaned).strip()


def _summarize(text: str, max_chars: int = 300) -> str:
    """Trim to a voice-friendly length for TTS."""
    text = re.sub(r"[#*_`\[\]]", "", text)
    text = re.sub(r"\n{2,}", "。", text)
    text = text.replace("\n", "，").strip()
    if len(text) <= max_chars:
        return text
    for sep in ("。", "，", ". ", ", "):
        cut = text[:max_chars].rfind(sep)
        if cut > max_chars // 3:
            return text[: cut + len(sep)].rstrip("，,") + "。"
    return text[:max_chars] + "……"


async def _notify_speak(text: str, device_id: str = ""):
    """POST result to /api/speak so the voice client hears it."""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"text": text}
            if device_id:
                payload["device_id"] = device_id
            async with session.post(_SPEAK_URL, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                logger.info("openclaw /api/speak response: %s", resp.status)
    except Exception as exc:
        logger.error("openclaw /api/speak failed: %s", exc)


async def delegate_auto(message: str, device_id: str = "") -> str:
    """Auto-route: async if keywords match, otherwise sync."""
    if should_async(message):
        return await delegate_async(message, device_id=device_id)
    return await delegate_sync(message)


async def delegate_sync(message: str) -> str:
    """Send *message* to OpenClaw and wait for the reply."""
    cmd = _build_cmd(message, _SYNC_TIMEOUT, fresh_session=True)
    logger.info("openclaw [sync] sending: %s", message[:80])

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_SYNC_TIMEOUT + 10
        )
    except asyncio.TimeoutError:
        logger.warning("openclaw [sync] timed out after %ds", _SYNC_TIMEOUT)
        return "OpenClaw 執行超時，任務可能仍在背景運行。"
    except Exception as exc:
        logger.error("openclaw [sync] failed: %s", exc)
        return f"OpenClaw 呼叫失敗：{exc}"

    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()[:300]
        logger.warning("openclaw [sync] exit=%d stderr=%s", proc.returncode, err)
        return f"OpenClaw 回傳錯誤：{err}" if err else "OpenClaw 執行失敗，請稍後再試。"

    result = _clean_output(stdout.decode(errors="replace"))
    logger.info("openclaw [sync] got %d chars", len(result))
    return result if result else "OpenClaw 執行完成但沒有回傳內容。"


async def delegate_async(message: str, device_id: str = "") -> str:
    """Launch OpenClaw in background; when done, push result via /api/speak."""
    cmd = _build_cmd(message, _ASYNC_TIMEOUT, fresh_session=True)
    logger.info("openclaw [async] launching: %s", message[:80])

    async def _background():
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_ASYNC_TIMEOUT + 10
            )
        except asyncio.TimeoutError:
            logger.warning("openclaw [async] timed out after %ds", _ASYNC_TIMEOUT)
            await _notify_speak("OpenClaw 背景任務超時了，可能還在處理中。", device_id)
            return
        except Exception as exc:
            logger.error("openclaw [async] background failed: %s", exc)
            await _notify_speak(f"OpenClaw 背景任務失敗：{exc}", device_id)
            return

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()[:200]
            logger.warning("openclaw [async] exit=%d", proc.returncode)
            await _notify_speak(f"OpenClaw 背景任務出錯了。{err[:50]}", device_id)
            return

        result = _clean_output(stdout.decode(errors="replace"))
        logger.info("openclaw [async] completed, got %d chars", len(result))

        if result:
            spoken = _summarize(result)
            await _notify_speak(f"OpenClaw 完成了。{spoken}", device_id)
        else:
            await _notify_speak("OpenClaw 背景任務完成了，但沒有回傳內容。", device_id)

    asyncio.create_task(_background())
    return "好的，已在背景處理中，完成後會語音通知你。"
