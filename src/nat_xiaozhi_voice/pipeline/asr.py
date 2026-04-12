"""FunASR local batch speech-to-text.

Supports both SenseVoiceSmall and Fun-ASR-Nano models via ``funasr.AutoModel``.
Fun-ASR-Nano requires ``trust_remote_code`` and its model directory must contain
``model.py``, ``ctc.py``, and ``tools/`` from the Fun-ASR GitHub repo.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from typing import List

import opuslib_next

logger = logging.getLogger(__name__)

SPECIAL_TAG_RE = re.compile(r"<\|[^|]+\|>")


def _filter_special_tags(text: str) -> str:
    return SPECIAL_TAG_RE.sub("", text).strip()


def _is_fun_asr_nano(model_dir: str) -> bool:
    """Detect Fun-ASR-Nano by checking for model.py in the model directory."""
    return os.path.isfile(os.path.join(model_dir, "model.py"))


class FunASRRecognizer:
    """Wraps ``funasr.AutoModel`` for whole-sentence batch ASR."""

    def __init__(self, model_dir: str):
        import torch
        from funasr import AutoModel

        force_cpu = os.environ.get("ASR_DEVICE", "").lower() == "cpu"
        device = "cpu" if force_cpu else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._is_nano = _is_fun_asr_nano(model_dir)

        if self._is_nano:
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            self._model = AutoModel(
                model=model_dir,
                trust_remote_code=True,
                remote_code=os.path.join(model_dir, "model.py"),
                vad_kwargs={"max_single_segment_time": 30000},
                disable_update=True,
                device=device,
            )
            logger.info("Fun-ASR-Nano loaded from %s (device=%s)", model_dir, device)
        else:
            self._model = AutoModel(
                model=model_dir,
                vad_kwargs={"max_single_segment_time": 30000},
                disable_update=True,
                hub="hf",
                device=device,
            )
            logger.info("FunASR model loaded from %s (device=%s)", model_dir, device)

    @staticmethod
    def decode_opus_to_pcm(opus_packets: List[bytes]) -> bytes:
        decoder = opuslib_next.Decoder(16000, 1)
        pcm_parts: list[bytes] = []
        for pkt in opus_packets:
            pcm_parts.append(decoder.decode(pkt, 960))
        return b"".join(pcm_parts)

    @staticmethod
    def _pcm_to_wav_path(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """Write raw 16-bit PCM to a temporary WAV file and return its path."""
        import struct
        import tempfile

        num_channels = 1
        sample_width = 2
        byte_rate = sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width
        data_size = len(pcm_bytes)

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE",
            b"fmt ", 16, 1, num_channels,
            sample_rate, byte_rate, block_align, 16,
            b"data", data_size,
        )
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            os.write(fd, header + pcm_bytes)
        finally:
            os.close(fd)
        return path

    async def recognize(self, pcm_bytes: bytes) -> str:
        start = time.time()
        if self._is_nano:
            wav_path = self._pcm_to_wav_path(pcm_bytes)
            try:
                result = await asyncio.to_thread(
                    self._model.generate,
                    input=[wav_path],
                    cache={},
                    language="中文",
                    batch_size=1,
                )
            finally:
                os.unlink(wav_path)
        else:
            result = await asyncio.to_thread(
                self._model.generate,
                input=pcm_bytes,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
        text = ""
        if result and len(result) > 0:
            raw = result[0].get("text", "")
            text = _filter_special_tags(raw)
        elapsed = time.time() - start
        logger.info("ASR %.3fs: %s", elapsed, text[:80] if text else "(empty)")
        return text
