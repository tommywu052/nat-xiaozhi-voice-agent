"""TTS backends: CosyVoice (local HTTP) and Edge TTS (Microsoft cloud).

Streams PCM, encodes to Opus, feeds AudioRateController.
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
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U00002600-\U000026ff"
    "\U0000fe00-\U0000fe0f"
    "\U0000200d"
    "]+",
    flags=re.UNICODE,
)


def _clean_for_tts(text: str) -> str:
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"<[^>]+>", "", text)  # strip XML/HTML-like tags
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"[`~]", "", text)
    text = re.sub(r"[()（）\[\]【】{}「」『』<>《》]", "", text)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)  # bullet lists
    text = re.sub(r"\n{2,}", "，", text)  # collapse double newlines
    text = re.sub(r"\n", "，", text)  # newlines to comma
    text = re.sub(r"，{2,}", "，", text)  # dedup commas
    return text.strip().strip("，")


def _pcm_to_opus_frames(
    pcm: bytes,
    sample_rate: int,
    opus_callback: Callable[[bytes], None],
) -> None:
    encoder = OpusEncoder(sample_rate, channels=1, frame_size_ms=60)
    frame_bytes = int(sample_rate * 1 * 60 / 1000 * 2)
    pcm_buf = bytearray(pcm)
    try:
        while len(pcm_buf) >= frame_bytes:
            frame = bytes(pcm_buf[:frame_bytes])
            del pcm_buf[:frame_bytes]
            encoder.encode_pcm_stream(frame, False, opus_callback)
        if pcm_buf:
            encoder.encode_pcm_stream(bytes(pcm_buf), True, opus_callback)
        else:
            encoder.encode_pcm_stream(b"", True, opus_callback)
    finally:
        encoder.close()


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
        text = _clean_for_tts(text)
        if not text:
            return

        encoder = OpusEncoder(sample_rate, channels=1, frame_size_ms=60)
        frame_bytes = int(sample_rate * 1 * 60 / 1000 * 2)
        pcm_buf = bytearray()

        payload = {
            "text": text,
            "spk_id": self.spk_id,
            "stream": True,
            "target_sr": sample_rate,
        }
        timeout = aiohttp.ClientTimeout(total=30, sock_read=15)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("CosyVoice API %d: %s", resp.status, body[:200])
                        return

                    async for chunk in resp.content.iter_any():
                        if not chunk:
                            continue
                        pcm_buf.extend(chunk)
                        while len(pcm_buf) >= frame_bytes:
                            frame = bytes(pcm_buf[:frame_bytes])
                            del pcm_buf[:frame_bytes]
                            encoder.encode_pcm_stream(frame, False, opus_callback)

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


class EdgeTTS:
    """Microsoft Edge TTS — free cloud service, no API key required."""

    def __init__(self, voice: str = "zh-TW-HsiaoChenNeural"):
        self.voice = voice
        logger.info("EdgeTTS initialized with voice=%s", voice)

    async def synthesize_stream(
        self,
        text: str,
        sample_rate: int,
        opus_callback: Callable[[bytes], None],
        is_last: bool = False,
    ) -> None:
        original = text
        text = _clean_for_tts(text)
        logger.info("EdgeTTS: %d chars (cleaned %d chars) sr=%d", len(original), len(text), sample_rate)
        if not text:
            logger.warning("EdgeTTS: text empty after cleaning, skipping")
            return

        try:
            import edge_tts
            import miniaudio

            logger.info("EdgeTTS: calling Microsoft TTS for %d chars...", len(text))
            communicate = edge_tts.Communicate(text, self.voice)
            mp3_chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_chunks.append(chunk["data"])

            if not mp3_chunks:
                logger.warning("EdgeTTS returned no audio for %d chars", len(text))
                return

            mp3_data = b"".join(mp3_chunks)
            decoded = await asyncio.to_thread(
                miniaudio.decode, mp3_data,
                nchannels=1,
                sample_rate=sample_rate,
                output_format=miniaudio.SampleFormat.SIGNED16,
            )
            pcm = bytes(decoded.samples)
            logger.info(
                "EdgeTTS decoded %d bytes MP3 -> %d bytes PCM (%.2fs @ %dHz)",
                len(mp3_data), len(pcm),
                decoded.num_frames / sample_rate, sample_rate,
            )
            _pcm_to_opus_frames(pcm, sample_rate, opus_callback)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("EdgeTTS error for text=%s", text[:40])
