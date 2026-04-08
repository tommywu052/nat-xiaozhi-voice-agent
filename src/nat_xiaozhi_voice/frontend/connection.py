"""Per-connection state and message handling for a single xiaozhi client."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from nat_xiaozhi_voice.pipeline.tts import CosyVoiceTTS
from nat_xiaozhi_voice.pipeline.vad import SileroVAD
from nat_xiaozhi_voice.utils.audio_codec import OpusEncoder, decode_opus_packet, create_opus_decoder, DECODE_FRAME_SAMPLES
from nat_xiaozhi_voice.utils.audio_rate_controller import AudioRateController, PRE_BUFFER_COUNT

if TYPE_CHECKING:
    from nat_xiaozhi_voice.pipeline.omni_e2e import Qwen3OmniE2E
    from nat_xiaozhi_voice.pipeline.tools import E2EToolExecutor

logger = logging.getLogger(__name__)


class ConnectionHandler:
    """Manages one WebSocket connection from an ESP32 / py-xiaozhi client."""

    def __init__(
        self,
        ws: WebSocket,
        *,
        vad: SileroVAD,
        asr: Any,
        tts: CosyVoiceTTS,
        agent_fn: Callable[[str, str], Awaitable[str]],
        welcome_msg: dict[str, Any],
        close_no_voice_seconds: int = 120,
        omni_e2e: Optional["Qwen3OmniE2E"] = None,
        omni_e2e_streaming: bool = False,
        tool_executor: Optional["E2EToolExecutor"] = None,
    ):
        self.ws = ws
        self._vad = vad
        self._asr = asr
        self._tts = tts
        self._agent_fn = agent_fn
        self._omni_e2e = omni_e2e
        self._omni_e2e_streaming = omni_e2e_streaming
        self._tool_executor = tool_executor
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
        self._processing_task: asyncio.Task | None = None

    # ── lifecycle ──────────────────────────────────────────────────────

    async def run(self):
        """Main loop — read messages until disconnect.

        Long-running operations (_process_voice, _run_agent_and_speak) are
        spawned as background tasks so that the receive loop stays responsive
        to WebSocket control frames (ping/pong) and abort messages.
        """
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
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
            await self._cleanup()

    async def _cleanup(self):
        self._closed = True
        self._rate_ctrl.stop()
        if self._rate_task and not self._rate_task.done():
            self._rate_task.cancel()
        if self._omni_e2e:
            self._omni_e2e.clear_history(self.session_id)  # type: ignore[union-attr]
        logger.info("Connection closed: device=%s session=%s", self.device_id, self.session_id)

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
            self._spawn_processing(self._process_voice())
        elif state == "detect":
            text = msg.get("text", "")
            if text:
                self._spawn_processing(self._run_agent_and_speak(text))

    def _spawn_processing(self, coro):
        """Run a long-running coroutine as a background task so the
        WebSocket receive loop stays responsive to pings and abort."""
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        self._processing_task = asyncio.create_task(self._safe_run(coro))

    async def _safe_run(self, coro):
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Processing task error")

    async def _handle_abort(self):
        self._collecting = False
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
            self._spawn_processing(self._process_voice())

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

        if self._omni_e2e:
            if self._omni_e2e_streaming:
                await self._process_voice_e2e_streaming(packets)
            else:
                await self._process_voice_e2e(packets)
            return

        pcm = self._asr.decode_opus_to_pcm(packets)
        text = await self._asr.recognize(pcm)

        cleaned = re.sub(r"[。．.，,、\s]+", "", text) if text else ""
        if not cleaned:
            logger.info("ASR result is empty/punctuation-only, skipping agent call")
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})
            return

        await self._send_json({"type": "stt", "text": text, "session_id": self.session_id})
        await self._run_agent_and_speak(text)

    async def _process_voice_e2e(self, packets: list[bytes]):
        """End-to-end: Audio → Qwen3-Omni (single call) → Text + Audio → Client."""
        from nat_xiaozhi_voice.pipeline.omni_e2e import resample_pcm

        pcm_in = self._omni_e2e.decode_opus_to_pcm(packets)  # type: ignore[union-attr]
        user_text, text_reply, pcm_out, src_sr = await self._omni_e2e.process_audio(pcm_in)  # type: ignore[union-attr]

        if not text_reply and not pcm_out:
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})
            return

        if user_text:
            await self._send_json({"type": "stt", "text": user_text, "session_id": self.session_id})

        if not pcm_out:
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})
            return

        if src_sr and src_sr != self.sample_rate:
            pcm_array = np.frombuffer(pcm_out, dtype=np.int16)
            pcm_array = resample_pcm(pcm_array, src_sr, self.sample_rate)
            pcm_out = pcm_array.tobytes()

        await self._send_json({"type": "tts", "state": "start", "session_id": self.session_id})
        self._tts_playing = True
        self._rate_ctrl.reset()
        self._rate_task = self._rate_ctrl.start(self._send_audio_frame)

        encoder = OpusEncoder(self.sample_rate, channels=1, frame_size_ms=60)
        frame_count = 0

        def _on_opus_frame(opus_bytes: bytes):
            nonlocal frame_count
            frame_count += 1
            self._rate_ctrl.add_audio(opus_bytes)

        encoder.encode_pcm_stream(pcm_out, True, _on_opus_frame)
        encoder.close()
        logger.info("E2E TTS encoded %d opus frames", frame_count)

        async def _send_stop():
            await self._send_json({"type": "tts", "state": "stop", "session_id": self.session_id})
            self._tts_playing = False

        self._rate_ctrl.add_message(_send_stop)

        async def _send_listen_start():
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})

        self._rate_ctrl.add_message(_send_listen_start)

    async def _process_voice_e2e_streaming(self, packets: list[bytes]):
        """Streaming E2E with async_chunk and hybrid tool-call support.

        Uses ``modalities=["text", "audio"]`` with ``stream=True`` so the
        Thinker/Talker/Code2Wav stages run in an async pipeline.  Audio chunks
        arrive as soon as Code2Wav produces them (~2 s TTFA).

        Tool-call detection:
          Text tokens stream ~1.3 s before audio.  If the model emits a
          ``<tool_call>`` tag, we abort the first stream, execute the tool,
          and make a second E2E call with the tool result (text-only input).
          Non-tool conversations are completely unaffected.
        """
        from nat_xiaozhi_voice.pipeline.omni_e2e import resample_pcm, _wav_to_pcm

        t0 = time.time()
        pcm_in = self._omni_e2e.decode_opus_to_pcm(packets)  # type: ignore[union-attr]
        if len(pcm_in) < 3200:
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})
            return

        audio_url = await self._omni_e2e.prepare_audio_url(pcm_in)  # type: ignore[union-attr]
        # ASR deferred: starts at first audio chunk or tool-call detection
        # so it doesn't steal GPU cycles from the critical E2E pipeline.
        asr_task: asyncio.Task | None = None

        messages = self._omni_e2e.build_audio_messages(audio_url, self.session_id)  # type: ignore[union-attr]

        tts_started = False
        text_reply = ""
        frame_count = 0
        audio_chunk_count = 0
        detected_sr = 0
        encoder: OpusEncoder | None = None
        text_ttft: float | None = None
        audio_ttft: float | None = None

        # Tool-call detection state
        maybe_tool_call = False
        maybe_tool_call_time: float | None = None
        tool_call_info: dict | None = None
        tool_call_confirmed_time: float | None = None
        post_tool_text = ""
        post_tool_audio_chunks = 0

        TOOL_CALL_TIMEOUT = 5.0
        ANEXT_TIMEOUT = 60.0

        def _on_opus_frame(opus_bytes: bytes):
            nonlocal frame_count
            frame_count += 1
            self._rate_ctrl.add_audio(opus_bytes)

        e2e_gen = self._omni_e2e.stream_e2e(messages=messages).__aiter__()  # type: ignore[union-attr]

        try:
            while True:
                if (
                    maybe_tool_call
                    and not tool_call_info
                    and maybe_tool_call_time
                    and (time.time() - maybe_tool_call_time) > TOOL_CALL_TIMEOUT
                ):
                    logger.warning(
                        "Tool call tag not closed within %.0fs, abandoning",
                        TOOL_CALL_TIMEOUT,
                    )
                    maybe_tool_call = False
                    maybe_tool_call_time = None
                    if asr_task is None:
                        asr_task = asyncio.create_task(self._omni_e2e.transcribe(audio_url))  # type: ignore[union-attr]
                    break

                try:
                    modality, data = await asyncio.wait_for(
                        anext(e2e_gen), timeout=ANEXT_TIMEOUT,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning("E2E stream chunk timeout (%.0fs), aborting", ANEXT_TIMEOUT)
                    if asr_task is None:
                        asr_task = asyncio.create_task(self._omni_e2e.transcribe(audio_url))  # type: ignore[union-attr]
                    break

                if modality == "text":
                    if text_ttft is None:
                        text_ttft = time.time() - t0
                    text_reply += data  # type: ignore[operator]

                    if tool_call_info:
                        post_tool_text += data  # type: ignore[operator]
                        continue

                    if self._tool_executor and not tts_started:
                        if not maybe_tool_call and "<tool_call>" in text_reply:
                            maybe_tool_call = True
                            maybe_tool_call_time = time.time()
                            if asr_task is None:
                                asr_task = asyncio.create_task(self._omni_e2e.transcribe(audio_url))  # type: ignore[union-attr]
                            logger.info("Possible tool call detected in E2E text stream")
                        if maybe_tool_call and "</tool_call>" in text_reply:
                            from nat_xiaozhi_voice.pipeline.tools import parse_tool_call
                            tool_call_info = parse_tool_call(text_reply)
                            if tool_call_info:
                                tool_call_confirmed_time = time.time()
                                logger.info(
                                    "Tool call confirmed at %.2fs: %s — NOT breaking, observing stream...",
                                    tool_call_confirmed_time - t0, tool_call_info,
                                )
                                if asr_task is None:
                                    asr_task = asyncio.create_task(self._omni_e2e.transcribe(audio_url))  # type: ignore[union-attr]
                            else:
                                maybe_tool_call = False
                                maybe_tool_call_time = None
                    continue

                if modality != "audio" or not data:
                    continue

                if tool_call_info:
                    post_tool_audio_chunks += 1
                    if post_tool_audio_chunks <= 3:
                        logger.info(
                            "Post-tool audio chunk #%d at %.2fs (size=%d bytes)",
                            post_tool_audio_chunks, time.time() - t0, len(data),
                        )
                    continue

                if maybe_tool_call:
                    continue

                if audio_ttft is None:
                    audio_ttft = time.time() - t0
                    if asr_task is None:
                        asr_task = asyncio.create_task(self._omni_e2e.transcribe(audio_url))  # type: ignore[union-attr]

                audio_chunk_count += 1

                raw: bytes = data  # type: ignore[assignment]
                chunk_sr = 0
                if raw[:4] == b"RIFF":
                    try:
                        raw, chunk_sr = _wav_to_pcm(raw)
                    except ValueError:
                        pass
                if chunk_sr:
                    detected_sr = chunk_sr
                elif detected_sr:
                    chunk_sr = detected_sr
                else:
                    chunk_sr = 24000

                if not tts_started:
                    await self._send_json({"type": "tts", "state": "start", "session_id": self.session_id})
                    self._tts_playing = True
                    self._rate_ctrl.reset()
                    self._rate_task = self._rate_ctrl.start(self._send_audio_frame)
                    encoder = OpusEncoder(self.sample_rate, channels=1, frame_size_ms=60)
                    tts_started = True

                if chunk_sr != self.sample_rate:
                    pcm_array = np.frombuffer(raw, dtype=np.int16)
                    if len(pcm_array) > 0:
                        pcm_array = resample_pcm(pcm_array, chunk_sr, self.sample_rate)
                        raw = pcm_array.tobytes()

                if encoder and raw:
                    encoder.encode_pcm_stream(raw, False, _on_opus_frame)

            if encoder:
                encoder.encode_pcm_stream(b"", True, _on_opus_frame)
                encoder.close()

        except Exception:
            logger.exception("E2E streaming error")

        # ── Observation log for tool-call experiment ───────────────────
        if tool_call_info and tool_call_confirmed_time:
            stream_end = time.time()
            logger.info(
                "=== TOOL STREAM OBSERVATION ===\n"
                "  tool detected at: %.2fs\n"
                "  stream ended at:  %.2fs\n"
                "  post-tool duration: %.2fs\n"
                "  post-tool text chunks: '%s'\n"
                "  post-tool audio chunks: %d\n"
                "  full text_reply: '%s'\n"
                "  ================================",
                tool_call_confirmed_time - t0,
                stream_end - t0,
                stream_end - tool_call_confirmed_time,
                post_tool_text[:200],
                post_tool_audio_chunks,
                text_reply[:300],
            )

        # ── Recover incomplete tool call (missing </tool_call>) ─────
        if not tool_call_info and "<tool_call>" in text_reply:
            from nat_xiaozhi_voice.pipeline.tools import parse_tool_call_lenient
            tool_call_info = parse_tool_call_lenient(text_reply)
            if tool_call_info:
                logger.info("Tool call recovered from incomplete tag: %s", tool_call_info)

        # ── Handle tool call ──────────────────────────────────────────
        if tool_call_info:
            t_tool_start = time.time()
            tool_detect_dur = t_tool_start - t0

            if asr_task is None:
                asr_task = asyncio.create_task(self._omni_e2e.transcribe(audio_url))  # type: ignore[union-attr]

            logger.info(
                "Executing tool %s(%s)",
                tool_call_info["name"], tool_call_info["arguments"],
            )
            tool_result = await self._tool_executor.execute(  # type: ignore[union-attr]
                tool_call_info["name"], tool_call_info["arguments"],
            )
            t_tool_done = time.time()
            logger.info(
                "Tool result (%.0f chars, exec=%.2fs): %s",
                len(tool_result), t_tool_done - t_tool_start, tool_result[:200],
            )

            user_text = ""
            if asr_task and asr_task.done():
                try:
                    user_text = asr_task.result() or ""
                except Exception:
                    pass
            if user_text:
                await self._send_json({"type": "stt", "text": user_text, "session_id": self.session_id})

            logger.info(
                "Tool timing: detect=%.2fs exec=%.2fs asr=%s → followup starts at %.2fs",
                tool_detect_dur, t_tool_done - t_tool_start,
                "ready" if user_text else "pending",
                time.time() - t0,
            )

            followup_msgs = self._omni_e2e.build_tool_followup_messages(  # type: ignore[union-attr]
                user_text or "（語音輸入）",
                tool_call_info["name"],
                tool_result,
                session_id=self.session_id,
            )
            await self._stream_e2e_followup(
                followup_msgs, t0, user_text or "",
            )

            if not user_text and asr_task and not asr_task.done():
                try:
                    user_text = await asyncio.wait_for(asr_task, timeout=10) or ""
                    if user_text:
                        await self._send_json({"type": "stt", "text": user_text, "session_id": self.session_id})
                except (asyncio.TimeoutError, Exception):
                    pass

            await self._omni_e2e.cleanup_audio_url(audio_url)  # type: ignore[union-attr]
            return

        # ── Normal flow (no tool call) ────────────────────────────────

        # Ensure ASR is started before cleanup (needs the WAV file)
        if asr_task is None:
            asr_task = asyncio.create_task(self._omni_e2e.transcribe(audio_url))  # type: ignore[union-attr]
        user_text = await asr_task

        await self._omni_e2e.cleanup_audio_url(audio_url)  # type: ignore[union-attr]

        if user_text:
            await self._send_json({"type": "stt", "text": user_text, "session_id": self.session_id})

        from nat_xiaozhi_voice.pipeline.tools import strip_tool_tags
        clean_reply = strip_tool_tags(text_reply) if text_reply else text_reply

        if tts_started:
            self._omni_e2e._append_turn(self.session_id, user_text or "", clean_reply)  # type: ignore[union-attr]

            elapsed = time.time() - t0
            logger.info(
                "StreamE2E done: text='%s' | text_ttft=%.2fs audio_ttft=%.2fs | "
                "%d audio chunks, %d opus frames, %.1fs total",
                clean_reply[:60],
                text_ttft or 0, audio_ttft or 0,
                audio_chunk_count, frame_count, elapsed,
            )

            async def _send_stop():
                await self._send_json({"type": "tts", "state": "stop", "session_id": self.session_id})
                self._tts_playing = False
            self._rate_ctrl.add_message(_send_stop)

            async def _send_listen():
                await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})
            self._rate_ctrl.add_message(_send_listen)
        else:
            logger.info("No audio produced, sending listen:start (text='%s')", clean_reply[:60] if clean_reply else "")
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})

    async def _stream_e2e_followup(
        self,
        messages: list[dict],
        t0: float,
        user_text: str,
    ):
        """Stream a follow-up E2E response after tool execution.

        No further tool-call detection (prevents infinite loops).
        Uses the same async_chunk streaming as the primary E2E path.
        """
        from nat_xiaozhi_voice.pipeline.omni_e2e import resample_pcm, _wav_to_pcm

        t_followup = time.time()
        tts_started = False
        text_reply = ""
        frame_count = 0
        audio_chunk_count = 0
        detected_sr = 0
        encoder: OpusEncoder | None = None
        text_ttft: float | None = None
        audio_ttft: float | None = None

        def _on_opus_frame(opus_bytes: bytes):
            nonlocal frame_count
            frame_count += 1
            self._rate_ctrl.add_audio(opus_bytes)

        async for modality, data in self._omni_e2e.stream_e2e(messages=messages):  # type: ignore[union-attr]
            if modality == "text":
                if text_ttft is None:
                    text_ttft = time.time() - t_followup
                text_reply += data  # type: ignore[operator]
                continue

            if modality != "audio" or not data:
                continue

            if audio_ttft is None:
                audio_ttft = time.time() - t_followup

            audio_chunk_count += 1

            raw: bytes = data  # type: ignore[assignment]
            chunk_sr = 0
            if raw[:4] == b"RIFF":
                try:
                    raw, chunk_sr = _wav_to_pcm(raw)
                except ValueError:
                    pass
            if chunk_sr:
                detected_sr = chunk_sr
            elif detected_sr:
                chunk_sr = detected_sr
            else:
                chunk_sr = 24000

            if not tts_started:
                await self._send_json({"type": "tts", "state": "start", "session_id": self.session_id})
                self._tts_playing = True
                self._rate_ctrl.reset()
                self._rate_task = self._rate_ctrl.start(self._send_audio_frame)
                encoder = OpusEncoder(self.sample_rate, channels=1, frame_size_ms=60)
                tts_started = True

            if chunk_sr != self.sample_rate:
                pcm_array = np.frombuffer(raw, dtype=np.int16)
                if len(pcm_array) > 0:
                    pcm_array = resample_pcm(pcm_array, chunk_sr, self.sample_rate)
                    raw = pcm_array.tobytes()

            if encoder and raw:
                encoder.encode_pcm_stream(raw, False, _on_opus_frame)

        if encoder:
            encoder.encode_pcm_stream(b"", True, _on_opus_frame)
            encoder.close()

        self._omni_e2e.append_tool_turn(  # type: ignore[union-attr]
            self.session_id, user_text, text_reply,
        )

        elapsed = time.time() - t0
        followup_dur = time.time() - t_followup
        logger.info(
            "Tool followup E2E done: text='%s' | "
            "followup_text_ttft=%.2fs followup_audio_ttft=%.2fs | "
            "%d audio chunks, %d opus frames, %.1fs followup, %.1fs total",
            text_reply[:60],
            text_ttft or 0, audio_ttft or 0,
            audio_chunk_count, frame_count, followup_dur, elapsed,
        )

        if tts_started:
            async def _send_stop():
                await self._send_json({"type": "tts", "state": "stop", "session_id": self.session_id})
                self._tts_playing = False
            self._rate_ctrl.add_message(_send_stop)

            async def _send_listen():
                await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})
            self._rate_ctrl.add_message(_send_listen)
        else:
            logger.warning("Tool followup produced no audio, sending text-only response")
            await self._send_json({"type": "tts", "state": "start", "session_id": self.session_id})
            await self._send_json({"type": "tts", "state": "stop", "session_id": self.session_id})
            await self._send_json({"type": "listen", "state": "start", "session_id": self.session_id})

    async def _run_agent_and_speak(self, user_text: str):
        """Invoke the NAT workflow agent, then stream TTS back."""
        import time as _time
        logger.info("Calling agent with: %s", user_text[:100])
        t0 = _time.time()
        try:
            reply = await self._agent_fn(user_text, self.session_id)
        except Exception:
            logger.exception("Agent error")
            reply = "抱歉，我遇到了一點問題。"
        logger.info("Agent replied in %.1fs: %s", _time.time() - t0,
                     (reply or "(empty)")[:120])

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

        logger.info("Starting TTS for %d chars …", len(reply))
        try:
            await self._tts.synthesize_stream(
                reply, self.sample_rate, _on_opus_frame, is_last=True,
            )
            logger.info("TTS done, %d opus frames generated", frame_count)
        except Exception:
            logger.exception("TTS error")

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
