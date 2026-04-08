"""Per-connection state and message handling for a single xiaozhi client.

Ported from xiaozhi-esp32-server's ``core/connection.py`` and
``core/providers/tts/base.py``, adapted for NAT's async architecture:

* **Streaming LLM** — tokens arrive via ``astream_events`` and are buffered.
* **Sentence splitting** — the buffer is flushed at punctuation boundaries.
  The *first* segment uses aggressive punctuation (including ``，``) so the
  user hears audio within ~2-3 s; subsequent segments split at sentence-ending
  punctuation only.
* **Chunked TTS** — each segment is synthesized independently and pushed to
  the audio rate-controller while the LLM continues streaming.
* **Per-device memory** — ``device_id`` is used as the LangGraph ``thread_id``
  so the same physical device resumes its conversation across reconnects.
  Memory is persisted to SQLite via ``AsyncSqliteSaver``.
* **Voice memory commands** — user can say "清除記憶" / "忘記我" / "重新開始"
  to wipe the device's conversation history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Optional

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from nat_xiaozhi_voice.pipeline.asr import FunASRRecognizer
from nat_xiaozhi_voice.pipeline.tts import CosyVoiceTTS, _clean_for_tts
from nat_xiaozhi_voice.pipeline.vad import SileroVAD
from nat_xiaozhi_voice.utils.audio_codec import decode_opus_packet, create_opus_decoder, DECODE_FRAME_SAMPLES
from nat_xiaozhi_voice.utils.audio_rate_controller import AudioRateController, PRE_BUFFER_COUNT

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence-splitting constants (ported from xiaozhi-esp32-server tts/base.py)
# ---------------------------------------------------------------------------
FIRST_SENTENCE_PUNCTS = frozenset("，,、。！？；：!?;:\n")
NORMAL_SENTENCE_PUNCTS = frozenset("。！？!?\n")
MAX_BUFFER_CHARS = 150
FIRST_SENTENCE_MIN_CHARS = 2

CLEAR_MEMORY_KEYWORDS = frozenset({
    "清除記憶", "清除记忆", "忘記我", "忘记我",
    "重新開始", "重新开始", "清空對話", "清空对话",
    "刪除記憶", "删除记忆",
})


class ConnectionHandler:
    """Manages one WebSocket connection from an ESP32 / py-xiaozhi client."""

    def __init__(
        self,
        ws: WebSocket,
        *,
        vad: SileroVAD,
        asr: FunASRRecognizer,
        tts: CosyVoiceTTS,
        agent_fn: Callable[[str, str], Awaitable[str]],
        agent_stream_fn: Optional[Callable[[str, str], AsyncIterator[str]]] = None,
        clear_memory_fn: Optional[Callable[[str], Awaitable[None]]] = None,
        welcome_msg: dict[str, Any],
        close_no_voice_seconds: int = 120,
    ):
        self.ws = ws
        self._vad = vad
        self._asr = asr
        self._tts = tts
        self._agent_fn = agent_fn
        self._agent_stream_fn = agent_stream_fn
        self._clear_memory_fn = clear_memory_fn
        self._welcome_msg = dict(welcome_msg)
        self._close_no_voice_s = close_no_voice_seconds

        self.session_id = str(uuid.uuid4().hex[:16])
        self.device_id: str = ""
        self.client_id: str = ""
        self.sample_rate: int = welcome_msg.get("audio_params", {}).get("sample_rate", 24000)

        # Per-connection VAD state
        self._vad_ctx = SileroVAD.new_conn_state()
        self._listen_mode = "auto"

        # Audio collection
        self._opus_packets: list[bytes] = []
        self._collecting = False

        # Rate controller for TTS output
        self._rate_ctrl = AudioRateController()
        self._rate_task: asyncio.Task | None = None

        # State flags
        self._tts_playing = False
        self._closed = False
        self._abort_requested = False

    # ── lifecycle ──────────────────────────────────────────────────────

    async def run(self):
        """Main loop — read messages until disconnect."""
        self._rate_task = self._rate_ctrl.start(self._send_audio_frame)
        try:
            while not self._closed:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.receive":
                    if "bytes" in msg and msg["bytes"]:
                        await self._on_binary(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        await self._on_text(msg["text"])
                elif msg["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Connection %s error", self.device_id)
        finally:
            await self._cleanup()

    async def _cleanup(self):
        self._closed = True
        self._rate_ctrl.stop()
        if self._rate_task and not self._rate_task.done():
            self._rate_task.cancel()
        logger.info("Connection closed: device=%s session=%s", self.device_id, self.session_id)

    # ── per-device memory ─────────────────────────────────────────────

    @property
    def memory_thread_id(self) -> str:
        """LangGraph thread_id: use device_id for per-device persistence."""
        if self.device_id and self.device_id != "unknown":
            return self.device_id
        return self.session_id

    async def _handle_clear_memory(self):
        """Wipe the device's conversation history and confirm via TTS."""
        tid = self.memory_thread_id
        if self._clear_memory_fn:
            try:
                await self._clear_memory_fn(tid)
                logger.info("Memory cleared for thread=%s (device=%s)", tid, self.device_id)
            except Exception:
                logger.exception("Failed to clear memory for thread=%s", tid)

        confirm = "好的，我已經忘記之前的對話了，我們重新開始吧！"
        await self._send_json({"type": "tts", "state": "start", "session_id": self.session_id})
        self._tts_playing = True
        self._rate_ctrl.reset()
        self._rate_task = self._rate_ctrl.start(self._send_audio_frame)

        frame_count = 0

        def _on_frame(opus_bytes: bytes):
            nonlocal frame_count
            frame_count += 1
            self._rate_ctrl.add_audio(opus_bytes)

        try:
            await self._tts.synthesize_stream(confirm, 24000, _on_frame)
        except Exception:
            logger.exception("TTS error during memory-clear confirmation")

        async def _send_stop():
            await self._send_json({"type": "tts", "state": "stop", "session_id": self.session_id})
            self._tts_playing = False

        self._rate_ctrl.add_message(_send_stop)

        async def _send_listen():
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})

        self._rate_ctrl.add_message(_send_listen)

    # ── text messages ─────────────────────────────────────────────────

    async def _on_text(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON: %s", raw[:100])
            return

        msg_type = msg.get("type", "")
        if msg_type == "hello":
            await self._handle_hello(msg)
        elif msg_type == "listen":
            await self._handle_listen(msg)
        elif msg_type == "abort":
            await self._handle_abort()
        elif msg_type == "ping":
            await self._send_json({"type": "pong"})
        else:
            logger.debug("Unknown message type: %s", msg_type)

    async def _handle_hello(self, msg: dict):
        audio_params = msg.get("audio_params")
        if audio_params:
            if "sample_rate" in audio_params:
                self.sample_rate = int(audio_params["sample_rate"])
            self._welcome_msg["audio_params"].update(audio_params)
        self._welcome_msg["session_id"] = self.session_id
        await self._send_json(self._welcome_msg)
        logger.info("Hello from device=%s, sample_rate=%d", self.device_id, self.sample_rate)

    async def _handle_listen(self, msg: dict):
        state = msg.get("state", "")
        self._listen_mode = msg.get("mode", self._listen_mode)

        if state == "start":
            self._reset_audio()
            self._collecting = True
        elif state == "stop":
            self._collecting = False
            await self._process_voice()
        elif state == "detect":
            text = msg.get("text", "")
            if text:
                await self._run_agent_and_speak(text)

    async def _handle_abort(self):
        self._collecting = False
        self._abort_requested = True
        self._rate_ctrl.reset()
        self._tts_playing = False
        await self._send_json({"type": "tts", "state": "stop", "session_id": self.session_id})

    # ── binary audio ──────────────────────────────────────────────────

    async def _on_binary(self, data: bytes):
        if not self._collecting:
            return
        # VAD check
        has_voice = self._vad.process_opus_packet(self._vad_ctx, data, self._listen_mode)
        if has_voice or self._listen_mode == "manual":
            self._opus_packets.append(data)
        if self._vad_ctx.get("voice_stop"):
            self._collecting = False
            self._vad_ctx["voice_stop"] = False
            await self._process_voice()

    # ── voice processing pipeline ─────────────────────────────────────

    def _reset_audio(self):
        self._opus_packets.clear()
        self._vad_ctx = SileroVAD.new_conn_state()

    async def _process_voice(self):
        if not self._opus_packets:
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})
            return
        packets = list(self._opus_packets)
        self._opus_packets.clear()

        pcm = self._asr.decode_opus_to_pcm(packets)
        text = await self._asr.recognize(pcm)
        if not text:
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})
            return

        await self._send_json({"type": "stt", "text": text, "session_id": self.session_id})
        await self._run_agent_and_speak(text)

    # ── sentence splitting (ported from xiaozhi-esp32-server) ─────────

    @staticmethod
    def _find_sentence_boundary(text: str, is_first: bool) -> int:
        """Return the index of the first sentence-splitting punctuation, or -1."""
        puncts = FIRST_SENTENCE_PUNCTS if is_first else NORMAL_SENTENCE_PUNCTS
        for i, ch in enumerate(text):
            if ch in puncts:
                return i
        return -1

    # ── streaming agent + chunked TTS (core pipeline) ─────────────────

    async def _run_agent_and_speak(self, user_text: str):
        """Stream LLM tokens → split into sentences → synthesize TTS per-segment.

        The producer task reads LLM tokens and pushes text segments onto an
        ``asyncio.Queue``.  The consumer task reads segments, calls TTS for
        each one, and pushes Opus frames to the rate-controller.  Because both
        are asyncio tasks, TTS synthesis for segment N happens concurrently
        with LLM token streaming for segment N+1.
        """
        self._abort_requested = False

        # Voice command: clear memory
        stripped = user_text.strip().rstrip("。！？!?.，,")
        if stripped in CLEAR_MEMORY_KEYWORDS:
            await self._handle_clear_memory()
            return

        if self._agent_stream_fn is None:
            await self._run_agent_and_speak_legacy(user_text)
            return

        # -- set up TTS output ------------------------------------------------
        await self._send_json({"type": "tts", "state": "start", "session_id": self.session_id})
        self._tts_playing = True
        self._rate_ctrl.reset()
        self._rate_task = self._rate_ctrl.start(self._send_audio_frame)

        frame_count = 0
        full_reply_parts: list[str] = []
        segment_queue: asyncio.Queue[str | None] = asyncio.Queue()
        t_start = time.time()

        def _on_opus_frame(opus_bytes: bytes):
            nonlocal frame_count
            frame_count += 1
            self._rate_ctrl.add_audio(opus_bytes)

        # -- producer: LLM stream → sentence segments --------------------------
        async def _llm_producer():
            text_buf: list[str] = []
            is_first_sentence = True
            t_first_token = None

            try:
                async for chunk in self._agent_stream_fn(user_text, self.memory_thread_id):
                    if self._abort_requested:
                        break
                    if t_first_token is None:
                        t_first_token = time.time()
                        logger.info(
                            "LLM first token in %.2fs",
                            t_first_token - t_start,
                        )
                    full_reply_parts.append(chunk)
                    text_buf.append(chunk)

                    buffered = "".join(text_buf)

                    if is_first_sentence and len(buffered) >= FIRST_SENTENCE_MIN_CHARS:
                        boundary = self._find_sentence_boundary(buffered, True)
                    else:
                        boundary = self._find_sentence_boundary(buffered, is_first_sentence)

                    if boundary != -1:
                        segment = buffered[: boundary + 1]
                        remaining = buffered[boundary + 1 :]
                        text_buf = [remaining] if remaining else []
                        is_first_sentence = False
                        cleaned = _clean_for_tts(segment)
                        if cleaned:
                            await segment_queue.put(cleaned)
                    elif len(buffered) > MAX_BUFFER_CHARS:
                        text_buf = []
                        is_first_sentence = False
                        cleaned = _clean_for_tts(buffered)
                        if cleaned:
                            await segment_queue.put(cleaned)

                # flush remaining text
                remaining = "".join(text_buf).strip()
                if remaining and not self._abort_requested:
                    cleaned = _clean_for_tts(remaining)
                    if cleaned:
                        await segment_queue.put(cleaned)
            except Exception:
                logger.exception("LLM streaming error")
                if not full_reply_parts:
                    try:
                        fallback = await self._agent_fn(user_text, self.memory_thread_id)
                        if fallback:
                            full_reply_parts.append(fallback)
                            cleaned = _clean_for_tts(fallback)
                            if cleaned:
                                await segment_queue.put(cleaned)
                    except Exception:
                        logger.exception("Fallback agent call also failed")
            finally:
                await segment_queue.put(None)

        # -- consumer: sentence segments → TTS → Opus -------------------------
        async def _tts_consumer():
            tts_sr = 24000
            seg_idx = 0
            while True:
                segment = await segment_queue.get()
                if segment is None or self._abort_requested:
                    break
                seg_idx += 1
                t0 = time.time()
                try:
                    await self._tts.synthesize_stream(segment, tts_sr, _on_opus_frame)
                except Exception:
                    logger.exception("TTS error for segment #%d (%d chars)", seg_idx, len(segment))
                logger.debug(
                    "TTS segment #%d done in %.2fs (%d chars)",
                    seg_idx, time.time() - t0, len(segment),
                )

        # -- run producer and consumer concurrently ----------------------------
        try:
            producer = asyncio.create_task(_llm_producer())
            consumer = asyncio.create_task(_tts_consumer())
            await producer
            await consumer
        except Exception:
            logger.exception("Streaming pipeline error")

        reply = "".join(full_reply_parts)
        elapsed = time.time() - t_start
        logger.info(
            "Agent reply (%d chars, %.2fs total): %s",
            len(reply), elapsed, reply[:120],
        )
        logger.info("TTS produced %d opus frames", frame_count)

        # -- signal playback done ----------------------------------------------
        if not self._abort_requested:
            async def _send_stop():
                await self._send_json({"type": "tts", "state": "stop", "session_id": self.session_id})
                self._tts_playing = False

            self._rate_ctrl.add_message(_send_stop)

            async def _send_listen_start():
                await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})

            self._rate_ctrl.add_message(_send_listen_start)
        else:
            self._tts_playing = False

    # ── non-streaming fallback ────────────────────────────────────────

    async def _run_agent_and_speak_legacy(self, user_text: str):
        """Non-streaming fallback: wait for full reply, then synthesize."""
        try:
            reply = await self._agent_fn(user_text, self.memory_thread_id)
        except Exception:
            logger.exception("Agent error")
            reply = "抱歉，我遇到了一點問題。"

        logger.info("Agent reply (%d chars): %s", len(reply) if reply else 0, (reply or "")[:120])

        if not reply:
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})
            return

        await self._send_json({"type": "tts", "state": "start", "session_id": self.session_id})
        self._tts_playing = True
        self._rate_ctrl.reset()
        self._rate_task = self._rate_ctrl.start(self._send_audio_frame)

        frame_count = 0

        def _on_opus_frame(opus_bytes: bytes):
            nonlocal frame_count
            frame_count += 1
            self._rate_ctrl.add_audio(opus_bytes)

        tts_sample_rate = 24000
        try:
            await self._tts.synthesize_stream(
                reply, tts_sample_rate, _on_opus_frame, is_last=True,
            )
        except Exception:
            logger.exception("TTS error")
        logger.info("TTS produced %d opus frames", frame_count)

        async def _send_stop():
            await self._send_json({"type": "tts", "state": "stop", "session_id": self.session_id})
            self._tts_playing = False

        self._rate_ctrl.add_message(_send_stop)

        async def _send_listen_start():
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})

        self._rate_ctrl.add_message(_send_listen_start)

    # ── WebSocket I/O ─────────────────────────────────────────────────

    async def _send_json(self, data: dict):
        if self._closed or self.ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            await self.ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            self._closed = True

    async def _send_audio_frame(self, opus_bytes: bytes):
        if self._closed or self.ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            await self.ws.send_bytes(opus_bytes)
        except Exception:
            self._closed = True
