"""Register the Xiaozhi Voice front end with NAT.

After building the NAT workflow (which populates ``_shared_agent_state``),
this module creates:

* ``_agent_fn``            – non-streaming, returns the full reply string.
* ``_agent_stream_fn``     – async-generator that yields text chunks.
* ``_clear_memory_fn``     – clears per-device conversation history.
* ``_clear_all_memory_fn`` – clears all devices' history.
* ``_list_memory_devices`` – lists device IDs that have stored history.

All agent calls use ``device_id`` as the LangGraph ``thread_id`` so that
the same physical device resumes its conversation across reconnects.
Memory is persisted to SQLite via ``AsyncSqliteSaver``.
"""

from __future__ import annotations

import logging

import aiosqlite

from nat.cli.register_workflow import register_front_end
from nat.data_models.config import Config

from nat_xiaozhi_voice.frontend.config import XiaozhiVoiceFrontEndConfig

logger = logging.getLogger(__name__)


def _extract_content(msg) -> str:
    content = msg.content
    if isinstance(content, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "".join(parts)
    return str(content)


@register_front_end(config_type=XiaozhiVoiceFrontEndConfig)
async def register_xiaozhi_voice_front_end(
    config: XiaozhiVoiceFrontEndConfig,
    full_config: Config,
):
    """Build the agent function via NAT WorkflowBuilder, then yield the front-end plugin."""
    from langchain_core.messages import HumanMessage
    from nat.builder.workflow_builder import WorkflowBuilder

    from nat_xiaozhi_voice.frontend.plugin import XiaozhiVoiceFrontEndPlugin

    async with WorkflowBuilder.from_config(full_config) as builder:
        workflow = await builder.build(entry_function=config.workflow_function)

        from nat_xiaozhi_voice.workflow.register import _shared_agent_state

        agent = _shared_agent_state.get("agent")
        db_path = _shared_agent_state.get("db_path")

        if agent is None:
            logger.error(
                "Compiled agent graph not found in _shared_agent_state; "
                "streaming and per-device memory will be unavailable."
            )

        # ── agent functions ───────────────────────────────────────────

        def _track_user_text(text: str):
            try:
                from nat_xiaozhi_voice.tools.openclaw_delegate import set_last_user_text
                set_last_user_text(text)
            except Exception:
                pass

        async def _agent_fn(user_text: str, device_id: str) -> str:
            """Non-streaming agent call with per-device conversation memory."""
            _track_user_text(user_text)
            if agent is not None:
                cfg = {"configurable": {"thread_id": device_id}}
                output = await agent.ainvoke(
                    {"messages": [HumanMessage(content=user_text)]}, cfg
                )
                return _extract_content(output["messages"][-1])
            async with workflow.run(user_text) as runner:
                result = await runner.result()
            return str(result) if not isinstance(result, str) else result

        async def _agent_stream_fn(user_text: str, device_id: str):
            """Async-generator: yields text chunks as the LLM streams tokens.

            Uses ``agent.astream()`` instead of ``astream_events`` for
            significantly lower overhead — astream yields graph-state
            updates directly without the heavy event-bus machinery.
            """
            _track_user_text(user_text)
            if agent is None:
                result = await _agent_fn(user_text, device_id)
                if result:
                    yield result
                return

            cfg = {"configurable": {"thread_id": device_id}}
            async for chunk in agent.astream(
                {"messages": [HumanMessage(content=user_text)]},
                cfg,
                stream_mode="messages",
            ):
                msg, metadata = chunk
                # Tool call logging
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        logger.info(
                            "Tool CALL: %s | input=%s",
                            tc.get("name", "unknown"),
                            str(tc.get("args", ""))[:200],
                        )
                    continue

                # Tool result logging
                msg_type = type(msg).__name__
                if msg_type == "ToolMessage":
                    logger.info(
                        "Tool DONE: %s | output=%d chars",
                        getattr(msg, "name", "unknown"),
                        len(str(getattr(msg, "content", ""))),
                    )
                    continue

                # LLM text token
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content:
                    langgraph_node = metadata.get("langgraph_node", "")
                    if langgraph_node == "assistant":
                        yield content

        # ── memory management functions ───────────────────────────────

        async def _clear_memory_fn(device_id: str) -> None:
            """Delete all checkpoint data for a single device."""
            if not db_path:
                return
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in await cursor.fetchall()]
                for table in tables:
                    try:
                        await db.execute(
                            f"DELETE FROM [{table}] WHERE thread_id = ?",
                            (device_id,),
                        )
                    except Exception:
                        pass
                await db.commit()
            logger.info("Cleared memory for device=%s (db=%s)", device_id, db_path)

        async def _clear_all_memory_fn() -> None:
            """Delete all checkpoint data for every device."""
            if not db_path:
                return
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in await cursor.fetchall()]
                for table in tables:
                    try:
                        await db.execute(f"DELETE FROM [{table}]")
                    except Exception:
                        pass
                await db.commit()
            logger.info("Cleared ALL memory (db=%s)", db_path)

        async def _list_memory_devices_fn() -> list[str]:
            """Return a list of device_ids that have stored conversation history."""
            if not db_path:
                return []
            async with aiosqlite.connect(db_path) as db:
                try:
                    cursor = await db.execute(
                        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
                    )
                    return [row[0] for row in await cursor.fetchall()]
                except Exception:
                    return []

        # ── assemble plugin ───────────────────────────────────────────

        plugin = XiaozhiVoiceFrontEndPlugin(
            full_config,
            _agent_fn,
            _agent_stream_fn,
            clear_memory_fn=_clear_memory_fn,
            clear_all_memory_fn=_clear_all_memory_fn,
            list_memory_devices_fn=_list_memory_devices_fn,
        )
        yield plugin
