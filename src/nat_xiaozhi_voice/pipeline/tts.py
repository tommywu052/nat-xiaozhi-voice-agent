"""CosyVoice streaming TTS.

Mirrors ``core.providers.tts.cosyvoice_local`` from xiaozhi-esp32-server.
Streams PCM from CosyVoice HTTP, encodes to Opus, feeds AudioRateController.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Callable

import aiohttp

from nat_xiaozhi_voice.utils.audio_codec import OpusEncoder

logger = logging.getLogger(__name__)

PUNCT_SPLIT_RE = re.compile(r"(?<=[。！？；\n])")
EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U0001f170-\U0001f251"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U00002600-\U000026ff"
    "]+",
    flags=re.UNICODE,
)


def _clean_for_tts(text: str) -> str:
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"[`~]", "", text)
    return text.strip()


class CosyVoiceTTS:
    """Streams PCM chunks from CosyVoice, encodes to Opus, calls back per frame."""

    def __init__(self, api_url: str = "http://127.0.0.1:50000/tts_stream",
                 spk_id: str = "default"):
        self.api_url = api_url
        self.spk_id = spk_id

    async def synthesize_stream(
        self,
        text: str,
        sample_rate: int,
        opus_callback: Callable[[bytes], None],
        is_last: bool = False,
    ) -> None:
        """Stream TTS for *text*, calling *opus_callback* with each Opus frame."""
        text = _clean_for_tts(text)
        if not text:
            logger.warning("TTS text empty after cleaning, skipping")
            return

        encoder = OpusEncoder(sample_rate, channels=1, frame_size_ms=60)
        frame_bytes = int(sample_rate * 1 * 60 / 1000 * 2)
        pcm_buf = bytearray()
        total_pcm = 0

        payload = {
            "text": text,
            "spk_id": self.spk_id,
            "stream": True,
            "target_sr": sample_rate,
        }
        logger.info("TTS request: url=%s sr=%d text='%s'", self.api_url, sample_rate, text[:60])
        timeout = aiohttp.ClientTimeout(total=60, sock_read=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, json=payload) as resp:
                    logger.info("TTS response status=%d content_type=%s", resp.status, resp.content_type)
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("CosyVoice API %d: %s", resp.status, body[:200])
                        return

                    async for chunk in resp.content.iter_any():
                        if not chunk:
                            continue
                        total_pcm += len(chunk)
                        pcm_buf.extend(chunk)
                        while len(pcm_buf) >= frame_bytes:
                            frame = bytes(pcm_buf[:frame_bytes])
                            del pcm_buf[:frame_bytes]
                            encoder.encode_pcm_stream(frame, False, opus_callback)

                    logger.info("TTS stream finished: %d bytes PCM, %d bytes remaining in buf",
                                total_pcm, len(pcm_buf))

                    if pcm_buf:
                        encoder.encode_pcm_stream(bytes(pcm_buf), True, opus_callback)
                        pcm_buf.clear()
                    else:
                        encoder.encode_pcm_stream(b"", True, opus_callback)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("CosyVoice stream error for text=%s", text[:40])
        finally:
            encoder.close()
