"""Qwen3-Omni ASR — uses the vLLM OpenAI-compatible API for speech-to-text.

The vLLM service exposes Qwen3-Omni's Thinker which accepts audio input
and returns text.  Audio is saved as a temporary WAV file on Windows and
referenced via a ``file://`` URL with the WSL mount prefix so the vLLM
process running inside WSL2 can read it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import struct
import tempfile
import time
import uuid
from typing import List

import aiohttp
import opuslib_next

logger = logging.getLogger(__name__)

_DEFAULT_ASR_PROMPT = "請將這段語音精確轉換為文字。只輸出辨識到的語音內容，不要加任何額外說明。"


def _pcm_to_wav_bytes(
    pcm: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Pack raw PCM int16 samples into an in-memory WAV container."""
    data_size = len(pcm)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * channels * sample_width))
    buf.write(struct.pack("<H", channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


def _win_to_wsl_path(win_path: str) -> str:
    """``C:\\Users\\foo\\bar.wav`` → ``/mnt/c/Users/foo/bar.wav``"""
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


class Qwen3OmniASR:
    """Speech-to-text via Qwen3-Omni's OpenAI-compatible chat completions API.

    Parameters
    ----------
    api_url : str
        Base URL of the vLLM server, e.g. ``http://localhost:8901/v1``.
    model : str
        Model identifier returned by ``/v1/models``.
    asr_prompt : str
        Instruction text sent alongside the audio.
    use_data_url : bool
        If *True*, embed the WAV as a ``data:audio/wav;base64,…`` URL
        instead of writing a temp file.  Avoids FS sharing but increases
        request payload size.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8901/v1",
        model: str = "Qwen3-Omni-30B-A3B-Instruct",
        asr_prompt: str = _DEFAULT_ASR_PROMPT,
        use_data_url: bool = False,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.asr_prompt = asr_prompt
        self.use_data_url = use_data_url
        self._tmp_dir: str | None = None
        logger.info(
            "Qwen3OmniASR ready: api=%s model=%s data_url=%s",
            self.api_url, self.model, self.use_data_url,
        )

    # ── Opus decode (same interface as FunASRRecognizer) ──────────────

    @staticmethod
    def decode_opus_to_pcm(opus_packets: List[bytes]) -> bytes:
        decoder = opuslib_next.Decoder(16000, 1)
        pcm_parts: list[bytes] = []
        for pkt in opus_packets:
            pcm_parts.append(decoder.decode(pkt, 960))
        return b"".join(pcm_parts)

    # ── ASR ────────────────────────────────────────────────────────────

    async def recognize(self, pcm_bytes: bytes) -> str:
        if len(pcm_bytes) < 3200:  # < 0.1s at 16 kHz 16-bit mono
            logger.info("Audio too short (%d bytes), skipping ASR", len(pcm_bytes))
            return ""

        t0 = time.time()
        wav_bytes = _pcm_to_wav_bytes(pcm_bytes)

        audio_url = await self._prepare_audio_url(wav_bytes)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": self.asr_prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        text = ""
        timeout = aiohttp.ClientTimeout(total=30, sock_read=20)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_url}/chat/completions", json=payload,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("Qwen3-Omni ASR HTTP %d: %s", resp.status, body[:300])
                        return ""
                    data = await resp.json()
                    text = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
        except Exception:
            logger.exception("Qwen3-Omni ASR request failed")
            return ""
        finally:
            await self._cleanup_audio_url(audio_url)

        elapsed = time.time() - t0
        logger.info("ASR(Qwen3-Omni) %.3fs: %s", elapsed, text[:80] if text else "(empty)")
        return text

    # ── audio URL helpers ─────────────────────────────────────────────

    async def _prepare_audio_url(self, wav_bytes: bytes) -> str:
        if self.use_data_url:
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            return f"data:audio/wav;base64,{b64}"

        # File-based: save WAV to temp dir, expose via WSL file:// URL
        if self._tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp(prefix="nat_asr_")
            logger.info("ASR temp dir: %s", self._tmp_dir)

        fname = f"asr_{uuid.uuid4().hex[:8]}.wav"
        win_path = os.path.join(self._tmp_dir, fname)
        await asyncio.to_thread(self._write_file, win_path, wav_bytes)

        wsl_path = _win_to_wsl_path(win_path)
        return f"file://{wsl_path}"

    async def _cleanup_audio_url(self, audio_url: str) -> None:
        if audio_url.startswith("file://"):
            wsl_path = audio_url[len("file://"):]
            # Convert WSL path back to Windows path for deletion
            if wsl_path.startswith("/mnt/"):
                parts = wsl_path.split("/", 4)  # ['', 'mnt', 'c', 'Users', ...]
                if len(parts) >= 4:
                    drive = parts[2].upper()
                    rest = parts[3] if len(parts) > 3 else ""
                    rest = "/".join(parts[3:]) if len(parts) > 3 else ""
                    win_path = f"{drive}:\\{rest}".replace("/", "\\")
                else:
                    win_path = wsl_path
            else:
                win_path = wsl_path
            try:
                await asyncio.to_thread(os.unlink, win_path)
            except OSError:
                pass

    @staticmethod
    def _write_file(path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)
