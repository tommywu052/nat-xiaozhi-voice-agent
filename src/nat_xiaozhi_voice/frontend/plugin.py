"""NAT FrontEndBase implementation for the Xiaozhi Voice Agent."""

from __future__ import annotations

import logging
import typing

import uvicorn

from nat.builder.front_end import FrontEndBase
from nat_xiaozhi_voice.frontend.config import XiaozhiVoiceFrontEndConfig
from nat_xiaozhi_voice.frontend.ws_server import XiaozhiWSServer

if typing.TYPE_CHECKING:
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)


class XiaozhiVoiceFrontEndPlugin(FrontEndBase[XiaozhiVoiceFrontEndConfig]):
    """Runs the Xiaozhi-compatible WebSocket voice server."""

    def __init__(self, full_config: "Config", agent_fn):
        super().__init__(full_config)
        self._agent_fn = agent_fn

    async def run(self):
        cfg = self.front_end_config
        server = XiaozhiWSServer(cfg, self._agent_fn)

        logger.info(
            "Starting Xiaozhi Voice Agent on %s:%d%s",
            cfg.host, cfg.port, cfg.ws_path,
        )

        uv_config = uvicorn.Config(
            server.app,
            host=cfg.host,
            port=cfg.port,
            log_level="info",
        )
        uv_server = uvicorn.Server(uv_config)
        try:
            await uv_server.serve()
        except KeyboardInterrupt:
            logger.info("Voice server interrupted")
