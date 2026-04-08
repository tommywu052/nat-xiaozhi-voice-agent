"""Register the ``explain_scene`` VLM camera tool with NAT.

YAML usage (unified with main LLM)::

    functions:
      explain_scene:
        _type: explain_scene
        camera_index: 0
        llm_name: main_llm                            # reuse LLM provider
        vlm_model: "nvidia/llama-3.2-nv-vision-instruct-11b"  # multimodal model

YAML usage (standalone)::

    functions:
      explain_scene:
        _type: explain_scene
        camera_index: 0
        vlm_api_key: "sk-..."
        vlm_base_url: "https://integrate.api.nvidia.com/v1"
        vlm_model: "nvidia/llama-3.2-nv-vision-instruct-11b"

    functions:
      voice_agent:
        tool_names:
          - explain_scene
"""

from __future__ import annotations

import logging

from pydantic import Field

from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ExplainSceneConfig(FunctionBaseConfig, name="explain_scene"):
    """VLM camera tool — captures a frame and describes the scene."""

    camera_index: int = Field(default=0, description="USB camera device index")
    llm_name: Optional[LLMRef] = Field(
        default=None,
        description="Reference a configured LLM (e.g. 'main_llm') to reuse its base_url and api_key",
    )
    vlm_api_key: str = Field(default="", description="API key (ignored when llm_name is set)")
    vlm_base_url: str = Field(default="", description="Base URL (ignored when llm_name is set)")
    vlm_model: str = Field(
        default="meta/llama-3.2-11b-vision-instruct",
        description="Vision-language model name (overrides the LLM's model_name)",
    )
    default_prompt: str = Field(
        default="用繁體中文描述畫面，20字以內",
        description="Default VLM prompt when no specific query is given",
    )


def _resolve_llm_config(builder: Builder, llm_name: str) -> tuple[str, str]:
    """Extract base_url and api_key from a named LLM config."""
    llm_cfg = builder.get_llm_config(llm_name)
    base_url = getattr(llm_cfg, "base_url", "") or ""
    api_key_field = getattr(llm_cfg, "api_key", None)
    if api_key_field is not None and hasattr(api_key_field, "get_secret_value"):
        api_key = api_key_field.get_secret_value() or ""
    elif isinstance(api_key_field, str):
        api_key = api_key_field
    else:
        api_key = ""
    return base_url, api_key


@register_function(config_type=ExplainSceneConfig)
async def explain_scene_tool(config: ExplainSceneConfig, builder: Builder):
    from nat_xiaozhi_voice.tools import vlm_camera

    if config.llm_name is not None:
        base_url, api_key = _resolve_llm_config(builder, config.llm_name)
        logger.info("explain_scene: reusing LLM '%s' (base_url=%s)", config.llm_name, base_url)
    else:
        base_url = config.vlm_base_url
        api_key = config.vlm_api_key

    vlm_camera.configure(
        camera_index=config.camera_index,
        vlm_api_key=api_key,
        vlm_base_url=base_url,
        vlm_model=config.vlm_model,
        default_prompt=config.default_prompt,
    )

    async def _explain_scene(query: str = "") -> str:
        """Use this tool to describe the current scene captured by the robot's
        camera.  Call it when the user asks 'what do you see', 'describe the
        surroundings', 'look around', or wants to see or verify something
        through vision.  You may pass an optional *query* to focus the
        description (e.g. 'count the people' or 'read the text on the sign').
        """
        return await vlm_camera.analyze_scene(query)

    logger.info(
        "explain_scene tool registered (camera=%d, model=%s)",
        config.camera_index, config.vlm_model,
    )

    try:
        yield FunctionInfo.from_fn(
            _explain_scene,
            description=(
                "Capture a photo with the robot's camera and describe the scene. "
                "Use when the user asks to see, look, describe surroundings, or verify something visually. "
                "Optional query parameter to focus the description."
            ),
        )
    finally:
        vlm_camera.release_camera()
        logger.info("explain_scene tool cleaned up")
