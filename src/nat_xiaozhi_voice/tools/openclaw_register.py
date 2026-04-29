"""Register the ``openclaw`` tool with NAT.

Single tool with ONE parameter (message). Routing between sync/async
is handled automatically by keyword matching in openclaw_delegate.py,
so the LLM only decides *whether* to call the tool.

YAML usage::

    functions:
      openclaw:
        _type: openclaw
        session_id: "voice-bridge"
        sync_timeout: 120
        async_timeout: 300
"""

from __future__ import annotations

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class OpenClawConfig(FunctionBaseConfig, name="openclaw"):
    """Config for the OpenClaw delegation tool."""

    session_id: str = Field(default="voice-bridge")
    sync_timeout: int = Field(default=120)
    async_timeout: int = Field(default=300)


@register_function(config_type=OpenClawConfig)
async def openclaw_tool(config: OpenClawConfig, builder: Builder):
    from nat_xiaozhi_voice.tools import openclaw_delegate

    openclaw_delegate.configure(
        session_id=config.session_id,
        sync_timeout=config.sync_timeout,
        async_timeout=config.async_timeout,
    )

    async def openclaw(message: str = "") -> str:
        """Delegate a complex task to OpenClaw AI Agent.

        Use when the user asks you to:
        - research a topic in depth
        - send something to WhatsApp or other apps
        - set up a scheduled task or reminder
        - generate a long report

        Parameters
        ----------
        message : str
            The task description in natural language.
        """
        if not message or not message.strip():
            message = openclaw_delegate._last_user_text
            logger.warning("openclaw: LLM sent empty message, fallback to: %s", message[:80])
        if not message or not message.strip():
            return "請告訴我你想要 OpenClaw 做什麼。"
        return await openclaw_delegate.delegate_auto(message)

    logger.info(
        "openclaw tool registered (session=%s, sync=%ds, async=%ds)",
        config.session_id, config.sync_timeout, config.async_timeout,
    )

    yield FunctionInfo.from_fn(
        openclaw,
        description=(
            "Delegate a complex task to OpenClaw AI Agent. "
            "Use for deep research, sending to WhatsApp, "
            "scheduling tasks, or generating reports. "
            "Just describe the task in message."
        ),
    )
    logger.info("openclaw tool cleaned up")
