"""Register the voice_agent workflow function with NAT.

For Phase 0-2 this is a thin wrapper around NAT's built-in ``tool_calling_agent``.
Phase 3 will replace this with a full LangGraph StateGraph implementation.
"""

from __future__ import annotations

import asyncio
import logging
import typing

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


class VoiceAgentWorkflowConfig(FunctionBaseConfig, name="voice_agent"):
    """YAML: ``functions.voice_agent._type: voice_agent``"""

    llm_name: LLMRef
    tool_names: list[str] = Field(default_factory=list)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    max_iterations: int = Field(default=5)
    verbose: bool = Field(default=True)


@register_function(config_type=VoiceAgentWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def voice_agent_workflow(config: VoiceAgentWorkflowConfig, builder: Builder):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    llm: BaseChatModel = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    tools = []
    for name in config.tool_names:
        tool = await builder.get_tool(name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        tools.append(tool)

    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False) if tools else llm

    async def _assistant(state: MessagesState):
        sys = SystemMessage(content=config.system_prompt)
        return {"messages": [await llm_with_tools.ainvoke([sys] + state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", _assistant)
    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges("assistant", tools_condition)
        graph.add_edge("tools", "assistant")
    else:
        graph.add_edge(START, "assistant")

    agent = graph.compile()

    async def _run(input_text: str) -> str:
        output = await agent.ainvoke({"messages": [HumanMessage(content=input_text)]})
        last_msg = output["messages"][-1]
        content = last_msg.content
        if isinstance(content, list):
            parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
            return "".join(parts)
        return str(content)

    yield _run
