"""Quick VLM test — send a web image to Nemotron-3-Nano-Omni (local vLLM).

Supports two modes:
  python test_vlm.py              # Test local vLLM (default)
  python test_vlm.py --cloud      # Test NVIDIA NIM cloud API
"""

import asyncio
import os
import re
import sys
import openai

LOCAL_BASE_URL = "http://localhost:8880/v1"
LOCAL_MODEL = "nemotron"

CLOUD_BASE_URL = "https://integrate.api.nvidia.com/v1"
CLOUD_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
CLOUD_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

IMAGE_URL = (
    "https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/graphic-cards/"
    "50-series/rtx-5090/geforce-rtx-5090-learn-more-og-1200x630.jpg"
)

VIDEO_URL = (
    "https://blogs.nvidia.com/wp-content/uploads/2023/04/"
    "nvidia-studio-itns-wk53-scene-in-omniverse-1280w.mp4"
)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


async def test_image(client: openai.AsyncOpenAI, model: str):
    """Test image understanding."""
    print(f"\n{'─'*60}")
    print(f"[Image Test] Model: {model}")
    print(f"Image: {IMAGE_URL}")
    print("Sending request...\n")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "用繁體中文描述這張圖片，50字以內"},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                ],
            }
        ],
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    result = _strip_thinking(response.choices[0].message.content or "")
    print(f"VLM Response ({len(result)} chars):\n{result}")
    print(f"\nModel used  : {response.model}")
    if response.usage:
        print(f"Token usage : prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")


async def test_text(client: openai.AsyncOpenAI, model: str):
    """Test text chat (reasoning mode)."""
    print(f"\n{'─'*60}")
    print(f"[Text Test] Model: {model}")
    print("Sending request...\n")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "用繁體中文回答：vLLM 是什麼？一句話簡述。"},
        ],
        temperature=0.6,
        max_tokens=256,
    )

    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    content = msg.content or ""

    if reasoning:
        print(f"Reasoning ({len(str(reasoning))} chars):\n{str(reasoning)[:200]}...")
    print(f"\nContent:\n{content}")


async def test_video(client: openai.AsyncOpenAI, model: str):
    """Test video understanding (Nano Omni native capability)."""
    print(f"\n{'─'*60}")
    print(f"[Video Test] Model: {model}")
    print(f"Video: {VIDEO_URL}")
    print("Sending request...\n")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what happens in this video in Traditional Chinese, 100 chars max."},
                    {"type": "video_url", "video_url": {"url": VIDEO_URL}},
                ],
            }
        ],
        temperature=0.6,
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    result = _strip_thinking(response.choices[0].message.content or "")
    print(f"Video Response ({len(result)} chars):\n{result}")


async def main():
    use_cloud = "--cloud" in sys.argv

    if use_cloud:
        client = openai.AsyncOpenAI(api_key=CLOUD_API_KEY, base_url=CLOUD_BASE_URL)
        model = CLOUD_MODEL
        print("Mode: NVIDIA NIM Cloud API")
    else:
        client = openai.AsyncOpenAI(api_key="not-needed", base_url=LOCAL_BASE_URL)
        model = LOCAL_MODEL
        print("Mode: Local vLLM (Nemotron-3-Nano-Omni)")

    await test_text(client, model)
    await test_image(client, model)

    if not use_cloud:
        await test_video(client, model)

    print(f"\n{'═'*60}")
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
