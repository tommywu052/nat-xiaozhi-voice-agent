"""Register the voice_agent workflow function with NAT.

Builds a LangGraph StateGraph with **AsyncSqliteSaver** so that
per-device conversation history is persisted to disk and survives
server restarts.  The compiled graph is exposed via
``_shared_agent_state`` so the front-end can use ``astream``
for token-level streaming.

History compression: when the accumulated messages exceed
COMPRESS_THRESHOLD, a dedicated *compress* graph node generates
a concise LLM summary of the older turns.  Because the summary
LLM call lives in a separate node (``compress``), its streaming
tokens are filtered out by the front-end (which only yields tokens
from ``langgraph_node == "assistant"``).

Shared state keys:
* ``agent``          – compiled LangGraph agent
* ``system_prompt``  – active system prompt string
* ``db_path``        – path to the SQLite memory database
"""

from __future__ import annotations

import asyncio
import logging
import os
import typing

import aiosqlite
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

if typing.TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "你是小智，來自台灣的 AI 語音助手。說話簡短精準，適合語音場景。"
    "當需要即時資訊時，請主動使用工具。每個工具最多調用一次。"
)

MEMORY_DB_PATH = os.environ.get("XIAOZHI_MEMORY_DB", "xiaozhi_memory.db")
MAX_HISTORY_MESSAGES = 50
COMPRESS_THRESHOLD = 36
KEEP_RECENT = 12
SUMMARY_REFRESH_GAP = 6

_shared_agent_state: dict = {}
_summary_cache: dict[str, tuple[int, str]] = {}


class VoiceAgentWorkflowConfig(FunctionBaseConfig, name="voice_agent"):
    """YAML: ``functions.voice_agent._type: voice_agent``"""

    llm_name: LLMRef
    tool_names: list[str] = Field(default_factory=list)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    max_iterations: int = Field(default=5)
    verbose: bool = Field(default=True)


@register_function(config_type=VoiceAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def voice_agent_workflow(config: VoiceAgentWorkflowConfig, builder: Builder):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import START, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    llm: BaseChatModel = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    tools = []
    for name in config.tool_names:
        tool = await builder.get_tool(name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        tools.append(tool)

    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False) if tools else llm

    # ── Custom state with summary field ──────────────────────────
    from langgraph.graph import MessagesState

    class AgentState(MessagesState):
        history_summary: str = ""

    # ── compress node ────────────────────────────────────────────
    async def _compress(state: AgentState, config: RunnableConfig):
        """Summarize old messages when history exceeds threshold.

        Runs as a separate graph node so that its LLM streaming tokens
        are tagged ``langgraph_node="compress"`` and get filtered out
        by the front-end (which only forwards ``"assistant"`` tokens).
        """
        messages = state["messages"]
        if len(messages) > MAX_HISTORY_MESSAGES:
            messages = messages[-MAX_HISTORY_MESSAGES:]

        if len(messages) <= COMPRESS_THRESHOLD:
            return {}

        old_part = messages[:-KEEP_RECENT]
        old_count = len(old_part)
        thread_id = config.get("configurable", {}).get("thread_id", "default")

        cached = _summary_cache.get(thread_id)
        if cached:
            cached_count, cached_summary = cached
            if abs(old_count - cached_count) < SUMMARY_REFRESH_GAP:
                return {"history_summary": cached_summary}

        conv_lines: list[str] = []
        for m in old_part:
            content = str(getattr(m, "content", ""))[:150]
            if not content:
                continue
            if isinstance(m, HumanMessage):
                conv_lines.append(f"User: {content}")
            elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                conv_lines.append(f"AI: {content}")

        if not conv_lines:
            return {}

        text_block = "\n".join(conv_lines[-30:])
        try:
            resp = await llm.ainvoke([
                HumanMessage(content=(
                    "Summarize the following conversation in Traditional Chinese, 200 chars max.\n"
                    "MUST keep: user's name, preferences, and key topics discussed.\n"
                    "EXCLUDE real-time data (weather, news, stock prices) — those must be re-fetched.\n"
                    "EXCLUDE specific times and dates — those change and must be re-queried.\n"
                    "Output ONLY the summary, no extra formatting:\n\n" + text_block
                ))
            ])
            summary = str(resp.content).strip()
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)
            summary = ""

        if summary:
            _summary_cache[thread_id] = (old_count, summary)
            logger.info(
                "History compressed: %d msgs -> summary(%d chars) + %d recent",
                len(messages), len(summary), KEEP_RECENT,
            )
            return {"history_summary": summary}
        return {}

    # ── assistant node ───────────────────────────────────────────
    async def _assistant(state: AgentState):
        """Main LLM call — uses compressed history when available."""
        messages = state["messages"]
        summary = state.get("history_summary", "")

        if len(messages) > MAX_HISTORY_MESSAGES:
            messages = messages[-MAX_HISTORY_MESSAGES:]

        if summary and len(messages) > COMPRESS_THRESHOLD:
            recent = messages[-KEEP_RECENT:]
            sys = SystemMessage(
                content=config.system_prompt
                + "\n\n[Previous conversation summary]\n"
                + summary
            )
            messages_for_llm = [sys] + list(recent)
        else:
            sys = SystemMessage(content=config.system_prompt)
            messages_for_llm = [sys] + list(messages)

        return {"messages": [await llm_with_tools.ainvoke(messages_for_llm)]}

    # ── Build graph ──────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("compress", _compress)
    graph.add_node("assistant", _assistant)

    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "compress")
        graph.add_edge("compress", "assistant")
        graph.add_conditional_edges("assistant", tools_condition)
        graph.add_edge("tools", "assistant")
    else:
        graph.add_edge(START, "compress")
        graph.add_edge("compress", "assistant")

    db_path = MEMORY_DB_PATH
    conn = await aiosqlite.connect(db_path)
    memory = AsyncSqliteSaver(conn)
    await memory.setup()

    agent = graph.compile(checkpointer=memory)

    _shared_agent_state["agent"] = agent
    _shared_agent_state["system_prompt"] = config.system_prompt
    _shared_agent_state["db_path"] = db_path
    logger.info(
        "Voice agent compiled (db=%s, max_history=%d, compress_at=%d, keep_recent=%d)",
        db_path, MAX_HISTORY_MESSAGES, COMPRESS_THRESHOLD, KEEP_RECENT,
    )

    async def _run(input_text: str) -> str:
        cfg = {"configurable": {"thread_id": "default"}}
        output = await agent.ainvoke({"messages": [HumanMessage(content=input_text)]}, cfg)
        last_msg = output["messages"][-1]
        content = last_msg.content
        if isinstance(content, list):
            parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
            return "".join(parts)
        return str(content)

    try:
        yield _run
    finally:
        await conn.close()
        logger.info("SQLite memory connection closed")
