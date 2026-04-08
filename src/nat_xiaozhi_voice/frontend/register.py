"""Register the Xiaozhi Voice front end with NAT."""

from __future__ import annotations

import logging

from nat.cli.register_workflow import register_front_end
from nat.data_models.config import Config

from nat_xiaozhi_voice.frontend.config import XiaozhiVoiceFrontEndConfig

logger = logging.getLogger(__name__)


@register_front_end(config_type=XiaozhiVoiceFrontEndConfig)
async def register_xiaozhi_voice_front_end(
    config: XiaozhiVoiceFrontEndConfig,
    full_config: Config,
):
    """Build the agent function via NAT WorkflowBuilder.from_config, then yield the front-end plugin."""
    from nat.builder.workflow_builder import WorkflowBuilder

    from nat_xiaozhi_voice.frontend.plugin import XiaozhiVoiceFrontEndPlugin

    async with WorkflowBuilder.from_config(full_config) as builder:
        workflow = await builder.build(entry_function=config.workflow_function)

        async def _agent_fn(user_text: str, session_id: str) -> str:
            async with workflow.run(user_text) as runner:
                result = await runner.result()
            if isinstance(result, str):
                return result
            return str(result)

        plugin = XiaozhiVoiceFrontEndPlugin(full_config, _agent_fn)
        yield plugin
