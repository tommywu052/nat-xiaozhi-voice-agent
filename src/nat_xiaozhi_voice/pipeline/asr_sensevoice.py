"""FunASR local batch speech-to-text.

Mirrors ``core.providers.asr.fun_local`` from xiaozhi-esp32-server.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import List

import opuslib_next

logger = logging.getLogger(__name__)

SPECIAL_TAG_RE = re.compile(r"<\|[^|]+\|>")


def _filter_special_tags(text: str) -> str:
    return SPECIAL_TAG_RE.sub("", text).strip()


class FunASRRecognizer:
    """Wraps ``funasr.AutoModel`` for whole-sentence batch ASR."""

    def __init__(self, model_dir: str):
        import torch
        from funasr import AutoModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    async def recognize(self, pcm_bytes: bytes) -> str:
        start = time.time()
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
