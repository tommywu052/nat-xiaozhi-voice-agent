"""End-to-end voice pipeline: Audio In → Qwen3-Omni (Thinker+Talker+Code2Wav) → Audio Out.

A single vLLM-Omni API call replaces the separate ASR → Agent LLM → TTS pipeline,
reducing Thinker invocations from 3× to 1× for lower end-to-end latency.

Architecture (single API call):
  User Opus → PCM → WAV → [Thinker: understand + reply] → [Talker: speech tokens]
      → [Code2Wav: PCM audio] → text + WAV → resample → Opus → Client
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import struct
import tempfile
import time
import uuid
from typing import AsyncGenerator, List

import aiohttp
import numpy as np
import opuslib_next

logger = logging.getLogger(__name__)

_ASR_PROMPT = "請將這段語音精確轉換為文字。只輸出辨識到的語音內容，不要加任何額外說明。"

_BASE64_JUNK_RE = None  # lazy-compiled in _strip_base64_tail

def _strip_base64_tail(text: str) -> str:
    """Remove trailing base64-encoded audio data that vLLM-Omni appends."""
    global _BASE64_JUNK_RE
    if _BASE64_JUNK_RE is None:
        import re
        _BASE64_JUNK_RE = re.compile(r"[A-Za-z0-9+/=]{20,}$")
    return _BASE64_JUNK_RE.sub("", text).rstrip()

THINKER_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 2048,
    "detokenize": True,
    "repetition_penalty": 1.2,
    "stop_token_ids": [151645],
}

TALKER_PARAMS = {
    "temperature": 0.9,
    "top_k": 50,
    "max_tokens": 4096,
    "detokenize": False,
    "repetition_penalty": 1.05,
    "stop_token_ids": [2150],
}

CODE2WAV_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 65536,
    "detokenize": True,
    "repetition_penalty": 1.1,
}


def _pcm_to_wav_bytes(
    pcm: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2,
) -> bytes:
    data_size = len(pcm)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
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
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    buf = io.BytesIO(wav_bytes)
    if buf.read(4) != b"RIFF":
        raise ValueError("Not a valid WAV")
    buf.read(4)
    if buf.read(4) != b"WAVE":
        raise ValueError("Not a valid WAV")
    sample_rate = 0
    pcm_data = b""
    while True:
        chunk_id = buf.read(4)
        if len(chunk_id) < 4:
            break
        chunk_size = struct.unpack("<I", buf.read(4))[0]
        if chunk_id == b"fmt ":
            fmt_data = buf.read(chunk_size)
            _, _, sample_rate = struct.unpack("<HHI", fmt_data[:8])
        elif chunk_id == b"data":
            pcm_data = buf.read(chunk_size)
        else:
            buf.read(chunk_size)
    return pcm_data, sample_rate


def resample_pcm(pcm_int16: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return pcm_int16
    float_data = pcm_int16.astype(np.float32) / 32768.0
    ratio = target_sr / orig_sr
    new_len = int(len(float_data) * ratio)
    indices = np.arange(new_len) / ratio
    indices = np.clip(indices, 0, len(float_data) - 1)
    idx_floor = indices.astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, len(float_data) - 1)
    frac = indices - idx_floor
    resampled = float_data[idx_floor] * (1 - frac) + float_data[idx_ceil] * frac
    return (resampled * 32768.0).clip(-32768, 32767).astype(np.int16)


class Qwen3OmniE2E:
    """True end-to-end voice conversation via Qwen3-Omni vLLM-Omni.

    One API call handles: audio understanding → text reasoning → speech synthesis.
    Maintains per-session conversation history for multi-turn context.
    """

    MAX_HISTORY_TURNS = 10

    def __init__(
        self,
        api_url: str = "http://localhost:8901/v1",
        model: str = "Qwen3-Omni-30B-A3B-Instruct",
        system_prompt: str = "You are a helpful assistant.",
        user_audio_prompt: str = "請聯這段語音並回覆。",
        *,
        base_system_prompt: str = "",
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.base_system_prompt = base_system_prompt or system_prompt
        self.user_audio_prompt = user_audio_prompt
        self._tmp_dir: str | None = None
        self._history: dict[str, list[dict]] = {}
        logger.info("Qwen3OmniE2E ready: api=%s model=%s", self.api_url, self.model)

    def _get_history(self, session_id: str) -> list[dict]:
        if session_id not in self._history:
            self._history[session_id] = []
        return self._history[session_id]

    def _append_turn(self, session_id: str, user_text: str, assistant_text: str):
        history = self._get_history(session_id)
        if user_text:
            history.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        if assistant_text:
            history.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
        self._trim_history(history)

    def append_tool_turn(self, session_id: str, user_text: str, assistant_reply: str):
        """Store a tool turn as a clean user+assistant pair.

        No metadata or tool result in the stored text — only the natural
        language reply — so the model won't regurgitate raw data when
        referencing history, and TTS stays clean.
        """
        self._append_turn(session_id, user_text, assistant_reply)

    def _trim_history(self, history: list[dict]):
        while len(history) > self.MAX_HISTORY_TURNS * 2:
            del history[:2]

    def clear_history(self, session_id: str):
        self._history.pop(session_id, None)

    @staticmethod
    def decode_opus_to_pcm(opus_packets: List[bytes]) -> bytes:
        decoder = opuslib_next.Decoder(16000, 1)
        return b"".join(decoder.decode(pkt, 960) for pkt in opus_packets)

    async def transcribe(self, audio_url: str) -> str:
        """Lightweight ASR-only call (Thinker only, no Talker/Code2Wav).

        Runs in parallel with ``process_audio`` using the same WAV file URL.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": _ASR_PROMPT},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        t0 = time.time()
        timeout = aiohttp.ClientTimeout(total=30, sock_read=20)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_url}/chat/completions", json=payload,
                ) as resp:
                    if resp.status != 200:
                        return ""
                    data = await resp.json()
                    text = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
        except Exception:
            logger.debug("Parallel ASR failed (non-critical)", exc_info=True)
            return ""
        logger.info("ASR(parallel) %.1fs: %s", time.time() - t0, text[:60])
        return text

    async def process_audio(self, pcm_bytes: bytes) -> tuple[str, str, bytes, int]:
        """Audio-in → (user_transcript, text_reply, raw_pcm_out, output_sample_rate).

        Runs E2E inference and a parallel ASR transcription so the caller gets
        both what the user said and the agent's audio reply.
        """
        if len(pcm_bytes) < 3200:
            logger.info("Audio too short (%d bytes), skipping", len(pcm_bytes))
            return "", "", b"", 0

        t0 = time.time()
        wav_bytes = _pcm_to_wav_bytes(pcm_bytes)
        audio_url = await self._save_wav(wav_bytes)

        user_content: list[dict] = [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
        ]
        if self.user_audio_prompt:
            user_content.append({"type": "text", "text": self.user_audio_prompt})

        payload = {
            "model": self.model,
            "sampling_params_list": [THINKER_PARAMS, TALKER_PARAMS, CODE2WAV_PARAMS],
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": user_content},
            ],
        }

        # Run E2E and parallel ASR concurrently
        e2e_task = asyncio.create_task(self._call_e2e(payload))
        asr_task = asyncio.create_task(self.transcribe(audio_url))

        text_reply, audio_wav = await e2e_task
        user_transcript = await asr_task

        # Clean up WAV after both tasks finish
        await self._cleanup_wav(audio_url)

        pcm_out, src_sr = b"", 0
        if audio_wav:
            try:
                pcm_out, src_sr = _wav_to_pcm(audio_wav)
            except Exception:
                logger.exception("Failed to parse audio WAV from E2E response")

        elapsed = time.time() - t0
        audio_dur = len(pcm_out) / (src_sr * 2) if src_sr else 0
        logger.info(
            "E2E(Qwen3-Omni) %.1fs: user='%s' reply='%s' audio=%.1fs@%dHz",
            elapsed, user_transcript[:40], text_reply[:60], audio_dur, src_sr,
        )
        return user_transcript, text_reply, pcm_out, src_sr

    # ── Streaming LLM (Thinker-only, for sentence-chunked TTS) ────────

    async def stream_llm_response(self, audio_url: str) -> AsyncGenerator[str, None]:
        """Stream text tokens from Thinker only (no Talker/Code2Wav).

        Sends audio input to the vLLM-Omni server as a regular (non-multi-stage)
        streaming chat/completions request.  The Talker and Code2Wav stages are
        not triggered—only the Thinker processes the audio and generates text.
        """
        user_content: list[dict] = [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
        ]
        if self.user_audio_prompt:
            user_content.append({"type": "text", "text": self.user_audio_prompt})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": user_content},
            ],
            "stream": True,
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 1024,
        }

        t0 = time.time()
        total_text = ""
        timeout = aiohttp.ClientTimeout(total=60, sock_read=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_url}/chat/completions", json=payload,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("Streaming LLM HTTP %d: %s", resp.status, body[:300])
                        return

                    buf = b""
                    async for raw_chunk in resp.content.iter_any():
                        buf += raw_chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            if line == b"data: [DONE]":
                                return
                            if not line.startswith(b"data: "):
                                continue
                            try:
                                data = json.loads(line[6:])
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    total_text += content
                                    yield content
                            except (json.JSONDecodeError, IndexError, KeyError):
                                continue
        except Exception:
            logger.exception("Streaming LLM failed")

        logger.info("StreamLLM %.1fs: '%s'", time.time() - t0, total_text[:80])

    # ── Message builders ──────────────────────────────────────────────

    def build_audio_messages(
        self, audio_url: str, session_id: str = "",
    ) -> list[dict]:
        """Build messages for an audio-input E2E call (with session history)."""
        user_content: list[dict] = [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
        ]
        if self.user_audio_prompt:
            user_content.append({"type": "text", "text": self.user_audio_prompt})

        history_msgs = list(self._get_history(session_id)) if session_id else []
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            *history_msgs,
            {"role": "user", "content": user_content},
        ]

    def build_tool_followup_messages(
        self,
        user_text: str,
        tool_name: str,
        tool_result: str,
        session_id: str = "",
    ) -> list[dict]:
        """Build messages for the follow-up E2E call after tool execution.

        History ends with the user question.  This method appends a
        combined user message containing the tool result + answer
        instructions so the model can synthesise a natural spoken reply.
        """
        prompt = (
            f"用戶問：「{user_text}」\n"
            f"你呼叫了工具 {tool_name}，查詢結果如下：\n{tool_result}\n\n"
            f"請嚴格根據上面的查詢結果回答用戶的問題，保持簡潔口語（2-3句話）。"
            f"只使用結果中的數據，不要推測或編造。不要輸出工具呼叫標籤。"
        )
        history_msgs = list(self._get_history(session_id)) if session_id else []
        return [
            {"role": "system", "content": [{"type": "text", "text": self.base_system_prompt}]},
            *history_msgs,
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

    # ── Streaming 3-stage E2E (async_chunk) ────────────────────────────

    async def stream_e2e(
        self,
        audio_url: str | None = None,
        session_id: str = "",
        *,
        messages: list[dict] | None = None,
    ) -> AsyncGenerator[tuple[str, str | bytes], None]:
        """Stream a full 3-stage (Thinker+Talker+Code2Wav) call with async_chunk.

        Requires the vLLM-Omni server to be started with an ``async_chunk: true``
        stage config (``qwen3_omni_single_gpu_async.yaml``).

        Either pass *audio_url* (builds default messages with session history)
        or provide *messages* directly (for tool-call follow-ups).

        Yields ``("text", token_str)`` or ``("audio", wav_bytes)`` tuples as
        SSE chunks arrive.  Audio bytes are base64-decoded from the SSE stream
        and typically contain WAV data (RIFF header + PCM).
        """
        if messages is None:
            if audio_url is None:
                raise ValueError("Either audio_url or messages must be provided")
            messages = self.build_audio_messages(audio_url, session_id)

        payload = {
            "model": self.model,
            "messages": messages,
            "sampling_params_list": [THINKER_PARAMS, TALKER_PARAMS, CODE2WAV_PARAMS],
            "modalities": ["text", "audio"],
            "stream": True,
        }

        t0 = time.time()
        text_total = ""
        audio_chunks = 0
        timeout = aiohttp.ClientTimeout(total=120, sock_read=60)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_url}/chat/completions", json=payload,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("StreamE2E HTTP %d: %s", resp.status, body[:300])
                        return

                    buf = b""
                    async for raw_chunk in resp.content.iter_any():
                        buf += raw_chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            if line == b"data: [DONE]":
                                return
                            if not line.startswith(b"data: "):
                                continue
                            try:
                                data = json.loads(line[6:])
                                modality = data.get("modality", "text")
                                delta = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                )
                                content = delta.get("content", "")
                                if not content:
                                    continue
                                if modality == "audio":
                                    audio_chunks += 1
                                    yield ("audio", base64.b64decode(content))
                                else:
                                    text_total += content
                                    yield ("text", content)
                            except (json.JSONDecodeError, IndexError, KeyError):
                                continue
        except Exception:
            logger.exception("StreamE2E failed")

        logger.info(
            "StreamE2E %.1fs: text='%s' audio_chunks=%d",
            time.time() - t0, text_total[:80], audio_chunks,
        )

    # ── Public WAV helpers ──────────────────────────────────────────────

    async def prepare_audio_url(self, pcm_bytes: bytes) -> str:
        """Save PCM as WAV and return a file:// URL accessible from WSL2."""
        wav_bytes = _pcm_to_wav_bytes(pcm_bytes)
        return await self._save_wav(wav_bytes)

    async def cleanup_audio_url(self, audio_url: str) -> None:
        """Delete the temporary WAV file."""
        await self._cleanup_wav(audio_url)

    # ── Internal helpers ────────────────────────────────────────────────

    async def _call_e2e(self, payload: dict) -> tuple[str, bytes]:
        """Core 3-stage API call. Returns (text_reply, wav_bytes)."""
        text_reply = ""
        audio_wav = b""
        timeout = aiohttp.ClientTimeout(total=120, sock_read=90)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_url}/chat/completions", json=payload,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("E2E API HTTP %d: %s", resp.status, body[:400])
                        return "", b""
                    data = await resp.json()
        except Exception:
            logger.exception("E2E API request failed")
            return "", b""

        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if msg.get("content"):
                text_reply = msg["content"]
            audio = msg.get("audio")
            if audio and isinstance(audio, dict) and audio.get("data"):
                audio_wav = base64.b64decode(audio["data"])

        return text_reply, audio_wav

    # ── WAV file helpers ────────────────────────────────────────────────

    async def _save_wav(self, wav_bytes: bytes) -> str:
        if self._tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp(prefix="nat_e2e_")
            logger.info("E2E temp dir: %s", self._tmp_dir)
        fname = f"e2e_{uuid.uuid4().hex[:8]}.wav"
        win_path = os.path.join(self._tmp_dir, fname)
        await asyncio.to_thread(self._write_file, win_path, wav_bytes)
        return f"file://{_win_to_wsl_path(win_path)}"

    async def _cleanup_wav(self, audio_url: str) -> None:
        if not audio_url.startswith("file://"):
            return
        wsl_path = audio_url[len("file://"):]
        if wsl_path.startswith("/mnt/"):
            parts = wsl_path.split("/", 4)
            if len(parts) >= 4:
                drive = parts[2].upper()
                rest = "/".join(parts[3:]) if len(parts) > 3 else ""
                win_path = f"{drive}:\\{rest}".replace("/", "\\")
            else:
                return
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
