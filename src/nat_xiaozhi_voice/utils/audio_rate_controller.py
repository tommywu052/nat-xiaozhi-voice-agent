"""Time-aligned audio rate controller for smooth TTS playback.

Mirrors ``AudioRateController`` from xiaozhi-esp32-server. Supports
a pre-buffer of N frames sent immediately to reduce first-packet latency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

PRE_BUFFER_COUNT = 5
DEFAULT_FRAME_DURATION_MS = 60


class AudioRateController:

    def __init__(self, frame_duration: int = DEFAULT_FRAME_DURATION_MS):
        self.frame_duration = frame_duration
        self._queue: deque[tuple[str, object]] = deque()
        self._play_position = 0
        self._start_ts: float | None = None
        self._task: asyncio.Task | None = None
        self._empty_event = asyncio.Event()
        self._empty_event.set()
        self._data_event = asyncio.Event()

    def reset(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._queue.clear()
        self._play_position = 0
        self._start_ts = None
        self._empty_event.set()
        self._data_event.clear()

    def add_audio(self, opus_packet: bytes):
        self._queue.append(("audio", opus_packet))
        self._empty_event.clear()
        self._data_event.set()

    def add_message(self, callback: Callable[[], Awaitable[None]]):
        self._queue.append(("message", callback))
        self._empty_event.clear()
        self._data_event.set()

    async def _drain(self, send_cb: Callable[[bytes], Awaitable[None]]):
        while self._queue:
            kind, payload = self._queue[0]
            if kind == "message":
                self._queue.popleft()
                await payload()  # type: ignore[operator]
            else:
                if self._start_ts is None:
                    self._start_ts = time.monotonic()
                elapsed_ms = (time.monotonic() - self._start_ts) * 1000
                if elapsed_ms < self._play_position:
                    wait_s = (self._play_position - elapsed_ms) / 1000
                    await asyncio.sleep(wait_s)
                self._queue.popleft()
                self._play_position += self.frame_duration
                await send_cb(payload)  # type: ignore[arg-type]
        self._empty_event.set()
        self._data_event.clear()

    def start(self, send_cb: Callable[[bytes], Awaitable[None]]) -> asyncio.Task:
        async def _loop():
            try:
                while True:
                    await self._data_event.wait()
                    await self._drain(send_cb)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("AudioRateController loop error")

        self._task = asyncio.create_task(_loop())
        return self._task

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()

    async def wait_empty(self):
        await self._empty_event.wait()
