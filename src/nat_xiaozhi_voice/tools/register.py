"""Register the ``explain_scene`` VLM camera tool with NAT.

Supports three camera sources:

* **Local camera** — captures from USB webcam on the NAT server.
* **Remote camera (HTTP)** — fetches image from a Robot camera server via HTTP.
* **Relay (built-in)** — Robot connects via WebSocket to NAT's ``/ws/robot``,
  no fixed Robot IP required.  Set ``relay_enabled: true`` in front-end config.

YAML usage (relay — recommended for dynamic Robot IP)::

    general:
      front_end:
        relay_enabled: true   # Robot connects to ws://host:port/ws/robot

    functions:
      explain_scene:
        _type: explain_scene
        llm_name: main_llm
        vlm_model: "google/gemma-4-31b-it"
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

    camera_index: int = Field(default=0, description="USB camera device index (local camera only)")
    remote_camera_url: str = Field(
        default="",
        description=(
            "URL of the remote Robot camera server (e.g. 'http://192.168.1.100:9903'). "
            "When set, images are fetched from the remote server instead of the local camera. "
            "The robot should run camera_server.py with --port matching this URL."
        ),
    )
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
        default="用繁體中文列出畫面中所有物品和人。每項一行，包含位置(左/中/右/前/後)、顏色、大小或形狀。只列清單，不要寫開頭總結。",
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


def _is_relay_enabled(builder: Builder) -> bool:
    """Check if relay_enabled is set in the front-end config."""
    try:
        wb = getattr(builder, "_workflow_builder", builder)
        gc = getattr(wb, "general_config", None)
        if gc is None:
            return False
        fe_cfg = getattr(gc, "front_end", None)
        if fe_cfg is None:
            return False
        return bool(getattr(fe_cfg, "relay_enabled", False))
    except Exception:
        return False


@register_function(config_type=ExplainSceneConfig)
async def explain_scene_tool(config: ExplainSceneConfig, builder: Builder):
    from nat_xiaozhi_voice.tools import vlm_camera

    if config.llm_name is not None:
        base_url, api_key = _resolve_llm_config(builder, config.llm_name)
        logger.info("explain_scene: reusing LLM '%s' (base_url=%s)", config.llm_name, base_url)
    else:
        base_url = config.vlm_base_url
        api_key = config.vlm_api_key

    use_relay = _is_relay_enabled(builder)

    vlm_camera.configure(
        camera_index=config.camera_index,
        vlm_api_key=api_key,
        vlm_base_url=base_url,
        vlm_model=config.vlm_model,
        default_prompt=config.default_prompt,
        remote_camera_url=config.remote_camera_url,
        use_relay=use_relay,
    )

    async def _explain_scene(query: str = "") -> str:
        """Use this tool to describe the current scene captured by the robot's
        camera.  Call it when the user asks 'what do you see', 'describe the
        surroundings', 'look around', or wants to see or verify something
        through vision.  You may pass an optional *query* to focus the
        description (e.g. 'count the people' or 'read the text on the sign').
        """
        return await vlm_camera.analyze_scene(query)

    if use_relay:
        logger.info(
            "explain_scene tool registered (RELAY built-in, vlm_model=%s)",
            config.vlm_model,
        )
    elif config.remote_camera_url:
        logger.info(
            "explain_scene tool registered (REMOTE=%s, vlm_model=%s)",
            config.remote_camera_url, config.vlm_model,
        )
    else:
        logger.info(
            "explain_scene tool registered (LOCAL camera=%d, vlm_model=%s)",
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
