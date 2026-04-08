"""Opus encode/decode utilities — mirrors xiaozhi-esp32-server's opus handling."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import opuslib_next
from opuslib_next import constants

logger = logging.getLogger(__name__)

DECODE_SAMPLE_RATE = 16_000
DECODE_CHANNELS = 1
DECODE_FRAME_SAMPLES = 960  # 60 ms @ 16 kHz


def create_opus_decoder(sample_rate: int = DECODE_SAMPLE_RATE,
                        channels: int = DECODE_CHANNELS) -> opuslib_next.Decoder:
    return opuslib_next.Decoder(sample_rate, channels)


def decode_opus_packet(decoder: opuslib_next.Decoder,
                       opus_packet: bytes,
                       frame_samples: int = DECODE_FRAME_SAMPLES) -> bytes:
    return decoder.decode(opus_packet, frame_samples)


class OpusEncoder:
    """PCM → Opus streaming encoder.

    Mirrors ``OpusEncoderUtils`` from xiaozhi-esp32-server with identical
    parameters so that downstream ESP32 / py-xiaozhi clients decode correctly.
    """

    def __init__(self, sample_rate: int, channels: int = 1, frame_size_ms: int = 60):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size_ms = frame_size_ms
        self.frame_size = (sample_rate * frame_size_ms) // 1000
        self.total_frame_size = self.frame_size * channels

        self.bitrate = 24_000
        self.complexity = 10

        self._buffer = np.array([], dtype=np.int16)
        self._encoder = opuslib_next.Encoder(
            sample_rate, channels, constants.APPLICATION_AUDIO,
        )
        self._encoder.bitrate = self.bitrate
        self._encoder.complexity = self.complexity
        self._encoder.signal = constants.SIGNAL_VOICE

    def reset(self):
        self._encoder.reset_state()
        self._buffer = np.array([], dtype=np.int16)

    def encode_pcm_stream(self, pcm_data: bytes, end_of_stream: bool,
                          callback: Callable[[bytes], None]) -> None:
        new_samples = np.frombuffer(pcm_data, dtype=np.int16)
        self._buffer = np.append(self._buffer, new_samples)

        offset = 0
        while offset <= len(self._buffer) - self.total_frame_size:
            frame = self._buffer[offset:offset + self.total_frame_size]
            encoded = self._encode_frame(frame)
            if encoded:
                callback(encoded)
            offset += self.total_frame_size
        self._buffer = self._buffer[offset:]

        if end_of_stream and len(self._buffer) > 0:
            last_frame = np.zeros(self.total_frame_size, dtype=np.int16)
            last_frame[:len(self._buffer)] = self._buffer
            encoded = self._encode_frame(last_frame)
            if encoded:
                callback(encoded)
            self._buffer = np.array([], dtype=np.int16)

    def _encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        try:
            return self._encoder.encode(frame.tobytes(), self.frame_size)
        except Exception:
            logger.exception("Opus encode error")
            return None

    def close(self):
        if hasattr(self, "_encoder") and self._encoder:
            del self._encoder
            self._encoder = None  # type: ignore[assignment]
