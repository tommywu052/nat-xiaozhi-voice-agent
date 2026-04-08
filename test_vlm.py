"""Quick VLM test — send a web image to NVIDIA NIM vision model."""

import asyncio
import os
import openai

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"
VLM_MODEL = "google/gemma-4-31b-it"

IMAGE_URL = (
    "https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/graphic-cards/"
    "50-series/rtx-5090/geforce-rtx-5090-learn-more-og-1200x630.jpg"
)


async def main():
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"Model : {VLM_MODEL}")
    print(f"Image : {IMAGE_URL}")
    print("Sending request...\n")

    response = await client.chat.completions.create(
        model=VLM_MODEL,
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
    )

    result = response.choices[0].message.content
    print(f"VLM Response ({len(result)} chars):\n{result}")
    print(f"\nModel used  : {response.model}")
    print(f"Token usage : prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
