"""Silero VAD — ONNX-based Voice Activity Detection.

Mirrors ``core.providers.vad.silero`` from xiaozhi-esp32-server.
Per-connection state is stored on a mutable context dict.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any

import numpy as np
import onnxruntime
import opuslib_next

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
DEFAULT_THRESHOLD_LOW = 0.2
DEFAULT_SILENCE_MS = 1000
FRAME_WINDOW_SIZE = 10
FRAME_WINDOW_THRESHOLD = 3
ONNX_CHUNK_SAMPLES = 512  # 512 samples = 32 ms @ 16 kHz
ONNX_CHUNK_BYTES = ONNX_CHUNK_SAMPLES * 2
OPUS_DECODE_SAMPLES = 960  # 60 ms @ 16 kHz


class SileroVAD:
    """Shared ONNX session — one per process; per-connection state is external."""

    def __init__(self, model_dir: str, threshold: float = DEFAULT_THRESHOLD,
                 threshold_low: float = DEFAULT_THRESHOLD_LOW,
                 silence_ms: int = DEFAULT_SILENCE_MS):
        model_path = os.path.join(model_dir, "src", "silero_vad", "data", "silero_vad.onnx")
        if not os.path.isfile(model_path):
            model_path = os.path.join(model_dir, "silero_vad.onnx")
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts,
        )
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.silence_ms = silence_ms
        logger.info("SileroVAD loaded from %s", model_path)

    @staticmethod
    def new_conn_state() -> dict[str, Any]:
        return {
            "decoder": opuslib_next.Decoder(16000, 1),
            "state": np.zeros((2, 1, 128), dtype=np.float32),
            "context": np.zeros((1, 64), dtype=np.float32),
            "audio_buffer": bytearray(),
            "voice_window": deque(maxlen=FRAME_WINDOW_SIZE),
            "last_is_voice": False,
            "have_voice": False,
            "voice_stop": False,
            "last_activity_ms": 0.0,
        }

    def process_opus_packet(self, ctx: dict[str, Any], opus_packet: bytes,
                            listen_mode: str = "auto") -> bool:
        """Returns True if voice is currently active. Sets ctx['voice_stop'] when silence detected."""
        if listen_mode == "manual":
            return True

        pcm = ctx["decoder"].decode(opus_packet, OPUS_DECODE_SAMPLES)
        ctx["audio_buffer"].extend(pcm)

        have_voice = False
        buf = ctx["audio_buffer"]
        while len(buf) >= ONNX_CHUNK_BYTES:
            chunk = bytes(buf[:ONNX_CHUNK_BYTES])
            del buf[:ONNX_CHUNK_BYTES]

            audio_i16 = np.frombuffer(chunk, dtype=np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
            audio_input = np.concatenate(
                [ctx["context"], audio_f32.reshape(1, -1)], axis=1,
            ).astype(np.float32)

            out, state = self._session.run(None, {
                "input": audio_input,
                "state": ctx["state"],
                "sr": np.array(16000, dtype=np.int64),
            })
            ctx["state"] = state
            ctx["context"] = audio_input[:, -64:]
            prob = out.item()

            if prob >= self.threshold:
                is_voice = True
            elif prob <= self.threshold_low:
                is_voice = False
            else:
                is_voice = ctx["last_is_voice"]
            ctx["last_is_voice"] = is_voice

            ctx["voice_window"].append(is_voice)
            have_voice = sum(ctx["voice_window"]) >= FRAME_WINDOW_THRESHOLD

            if ctx["have_voice"] and not have_voice:
                silence_dur = time.time() * 1000 - ctx["last_activity_ms"]
                if silence_dur >= self.silence_ms:
                    ctx["voice_stop"] = True

            if have_voice:
                ctx["have_voice"] = True
                ctx["last_activity_ms"] = time.time() * 1000

        return have_voice
