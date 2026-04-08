"""Standalone tool implementations for E2E voice pipeline tool calling.

Provides lightweight, async-callable tools invoked directly during Qwen3-Omni
E2E streaming—bypassing NAT/LangChain.  When Qwen3-Omni determines that a
user request requires a tool, it emits ``<tool_call>...</tool_call>`` XML tags
in its text stream.  The E2E pipeline intercepts these, executes the matching
tool here, and re-invokes the E2E stream with the tool result.

Tool prompt format follows Qwen3's native ``<tools>`` chat-template so the
model's function-calling behaviour is maximally reliable.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import zoneinfo
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ── Tool definitions (OpenAI-style, rendered into <tools> XML) ─────────

TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": (
                "僅用於查詢現在的日期和時間（幾點鐘、今天幾號、星期幾）。"
                "嚴禁用於天氣問題！「明天天氣」「後天下雨嗎」是天氣問題，必須用 get_weather。"
                "只有純粹問時間日期時才用此工具。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查詢天氣、氣溫、降雨、風向等氣象資訊。"
                "只要問題包含「天氣、下雨、溫度、氣溫、帶傘」任何一個關鍵字，都必須用此工具。"
                "包含「明天天氣」「後天下雨嗎」「今天溫度」也都用此工具，不要用 current_datetime。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名稱，例如「台北」「上海」「紐約」",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lunar",
            "description": (
                "查詢農曆日期、干支、生肖、節氣、黃曆宜忌。"
                "適用場景：農曆幾號、節氣、黃曆、今天宜忌。"
                "你的農曆知識是錯的，必須用此工具查詢，不可猜測。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "日期，格式 YYYY-MM-DD（選填，不提供則查今天）",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": (
                "在維基百科搜尋人物、歷史、知識等資訊。"
                "適用場景：查詢某人是誰、某事件、某概念的百科資訊。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "搜尋關鍵字或問題",
                    },
                },
                "required": ["question"],
            },
        },
    },
]


def build_tool_prompt_section() -> str:
    """Build the ``# Tools`` section using Qwen3's native chat-template format.

    The resulting text is appended to the persona system prompt so that
    Qwen3-Omni can emit ``<tool_call>`` tags when it decides to use a tool.
    """
    tool_json_lines = "\n".join(
        json.dumps(td, ensure_ascii=False) for td in TOOL_DEFS
    )
    return (
        "\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{tool_json_lines}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": "function_name", "arguments": {"arg1": "value1"}}\n'
        "</tool_call>\n\n"
        "重要規則：\n"
        "- 你沒有即時資訊。需要時必須選擇正確的工具：\n"
        "  * 純粹問時間/日期/星期/幾點 → current_datetime\n"
        "  * 涉及天氣/氣溫/下雨/帶傘（含「明天天氣」「後天下雨」） → get_weather（絕對不用 current_datetime）\n"
        "  * 農曆/節氣/黃曆/宜忌 → get_lunar（你的農曆知識是錯的，必須查詢）\n"
        "  * 人物/歷史/百科 → wiki_search\n"
        "- 需要工具時就直接輸出 <tool_call> 標籤，不要說「讓我幫你查」或「建議你使用某工具」。\n"
        "- 絕對不要在回覆中提及工具名稱（如 get_weather、current_datetime 等），用戶不需要知道工具的存在。\n"
        "- 先檢查對話歷史：如果你之前已回答過相同問題（如日期、天氣、農曆），"
        "直接引用歷史中的答案，不需要重新呼叫工具。\n"
        "- 閒聊、打招呼等不需要工具的場景，直接回答即可。\n"
        "- 呼叫工具時 **只** 輸出 <tool_call>…</tool_call>，不要附加其他文字。\n"
    )


# ── Tool-call detection / parsing ──────────────────────────────────────

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract the first ``<tool_call>`` JSON from *text*.

    Returns ``{"name": str, "arguments": dict}`` or *None*.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        if "name" in data:
            return {
                "name": data["name"],
                "arguments": data.get("arguments", {}),
            }
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool_call JSON: %s", m.group(1)[:200])
    return None


_KNOWN_TOOL_NAMES = {td["function"]["name"] for td in TOOL_DEFS}


def parse_tool_call_lenient(text: str) -> dict[str, Any] | None:
    """Try to parse a tool call even when ``</tool_call>`` is missing.

    Handles nested braces like ``{"name": "get_lunar", "arguments": {}}``.
    Also attempts to fix truncated JSON by appending closing braces.
    """
    result = parse_tool_call(text)
    if result:
        return result

    idx = text.find("<tool_call>")
    if idx == -1:
        return None
    content = text[idx + 11:].strip()

    brace_start = content.find("{")
    if brace_start == -1:
        logger.warning("Lenient parse: no '{' found after <tool_call>: %s", content[:100])
        return None

    snippet = content[brace_start:]

    # Try: find last '}' and parse from first '{' to last '}'
    brace_end = snippet.rfind("}")
    if brace_end > 0:
        candidate = snippet[: brace_end + 1]
        try:
            data = json.loads(candidate)
            if "name" in data and data["name"] in _KNOWN_TOOL_NAMES:
                return {"name": data["name"], "arguments": data.get("arguments", {})}
        except json.JSONDecodeError:
            pass

    # Try appending missing closing braces (truncated output)
    for suffix in ["}", "}}", "}}}"]:
        try:
            data = json.loads(snippet.rstrip() + suffix)
            if "name" in data and data["name"] in _KNOWN_TOOL_NAMES:
                logger.info("Lenient parse recovered with suffix '%s'", suffix)
                return {"name": data["name"], "arguments": data.get("arguments", {})}
        except json.JSONDecodeError:
            continue

    logger.warning("Lenient parse failed for: %s", snippet[:200])
    return None


def has_tool_call_start(text: str) -> bool:
    """Return *True* if *text* contains the opening ``<tool_call>`` tag."""
    return "<tool_call>" in text


_STRIP_TOOL_RE = re.compile(r"<tool_call>.*", re.DOTALL)


def strip_tool_tags(text: str) -> str:
    """Remove any ``<tool_call>`` fragment (complete or partial) from *text*."""
    return _STRIP_TOOL_RE.sub("", text).rstrip()


# ── Tool executor ──────────────────────────────────────────────────────

class E2EToolExecutor:
    """Executes tools called by Qwen3-Omni during E2E streaming."""

    def __init__(
        self,
        *,
        weather_api_host: str = "",
        weather_api_key: str = "",
        weather_default_location: str = "台北",
    ):
        self._weather_host = weather_api_host
        self._weather_key = weather_api_key
        self._weather_default = weather_default_location

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        logger.info("Executing tool: %s(%s)", tool_name, arguments)
        try:
            if tool_name == "current_datetime":
                return self._current_datetime()
            elif tool_name == "get_weather":
                loc = arguments.get("location", self._weather_default)
                return await self._get_weather(loc or self._weather_default)
            elif tool_name == "get_lunar":
                return self._get_lunar(arguments.get("date"))
            elif tool_name == "wiki_search":
                return await self._wiki_search(arguments.get("question", ""))
            else:
                return f"未知的工具: {tool_name}"
        except Exception as exc:
            logger.exception("Tool execution failed: %s", tool_name)
            return f"工具執行失敗: {exc}"

    # ── current_datetime ────────────────────────────────────────────

    @staticmethod
    def _current_datetime() -> str:
        tz = zoneinfo.ZoneInfo("Asia/Taipei")
        now = datetime.datetime.now(tz)
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        return (
            f"目前時間: {now.strftime('%Y年%m月%d日 %H:%M:%S')} "
            f"星期{weekdays[now.weekday()]} "
            f"時區: Asia/Taipei (UTC{now.strftime('%z')})"
        )

    # ── get_weather (qweather REST API) ─────────────────────────────

    async def _get_weather(self, location: str) -> str:
        if not self._weather_host or not self._weather_key:
            return f"天氣服務未設定，無法查詢 {location} 的天氣。"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        timeout = aiohttp.ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            geo_url = (
                f"https://{self._weather_host}/geo/v2/city/lookup"
                f"?key={self._weather_key}&location={location}&lang=zh"
            )
            async with session.get(geo_url) as resp:
                if resp.status != 200:
                    return f"查詢城市 {location} 失敗 (HTTP {resp.status})"
                geo_data = await resp.json()

            locations = geo_data.get("location", [])
            if not locations:
                return f"找不到城市: {location}"

            city = locations[0]
            city_id = city["id"]
            city_name = city["name"]

            now_url = (
                f"https://{self._weather_host}/v7/weather/now"
                f"?key={self._weather_key}&location={city_id}&lang=zh"
            )
            async with session.get(now_url) as resp:
                weather_data = await resp.json() if resp.status == 200 else {}

            now_info = weather_data.get("now", {})
            result = f"{city_name} 即時天氣:\n"
            if now_info:
                result += (
                    f"天氣: {now_info.get('text', '未知')}\n"
                    f"氣溫: {now_info.get('temp', '?')}°C "
                    f"(體感 {now_info.get('feelsLike', '?')}°C)\n"
                    f"濕度: {now_info.get('humidity', '?')}%\n"
                    f"風向: {now_info.get('windDir', '?')} "
                    f"{now_info.get('windScale', '?')}級\n"
                )

            forecast_url = (
                f"https://{self._weather_host}/v7/weather/3d"
                f"?key={self._weather_key}&location={city_id}&lang=zh"
            )
            async with session.get(forecast_url) as resp:
                forecast_data = await resp.json() if resp.status == 200 else {}

            daily = forecast_data.get("daily", [])
            if daily:
                result += "\n未來預報:\n"
                for day in daily[:3]:
                    result += (
                        f"  {day.get('fxDate', '')}: "
                        f"{day.get('textDay', '')} "
                        f"{day.get('tempMin', '')}~{day.get('tempMax', '')}°C\n"
                    )

            return result

    # ── get_lunar (cnlunar) ─────────────────────────────────────────

    @staticmethod
    def _get_lunar(date_str: str | None = None) -> str:
        try:
            import cnlunar
        except ImportError:
            return "農曆查詢功能未安裝（需要 cnlunar 套件）"

        if date_str:
            try:
                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return "日期格式錯誤，請使用 YYYY-MM-DD 格式"
        else:
            dt = datetime.datetime.now()

        lunar = cnlunar.Lunar(dt, godType="8char")
        result = (
            f"農曆: {lunar.lunarYearCn}年{lunar.lunarMonthCn[:-1]}{lunar.lunarDayCn}\n"
            f"干支: {lunar.year8Char}年 {lunar.month8Char}月 {lunar.day8Char}日\n"
            f"生肖: 屬{lunar.chineseYearZodiac}\n"
            f"星座: {lunar.starZodiac}\n"
            f"節氣: {lunar.todaySolarTerms}\n"
        )
        holidays = ",".join(
            filter(
                None,
                (
                    lunar.get_legalHolidays(),
                    lunar.get_otherHolidays(),
                    lunar.get_otherLunarHolidays(),
                ),
            )
        )
        if holidays:
            result += f"節日: {holidays}\n"
        result += (
            f"宜: {'、'.join(lunar.goodThing[:5])}\n"
            f"忌: {'、'.join(lunar.badThing[:5])}\n"
        )
        return result

    # ── wiki_search (Wikipedia REST API) ────────────────────────────

    @staticmethod
    async def _wiki_search(question: str) -> str:
        if not question:
            return "搜尋關鍵字不能為空"

        timeout = aiohttp.ClientTimeout(total=10)
        headers = {
            "User-Agent": "NATXiaozhiVoiceAgent/1.0 (https://github.com; contact@example.com)",
            "Accept": "application/json",
        }
        url = (
            "https://zh.wikipedia.org/w/api.php"
            "?action=query&list=search&format=json&utf8=1&srlimit=2"
            f"&srsearch={question}"
        )

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return f"維基百科搜尋失敗 (HTTP {resp.status})"
                data = await resp.json()

        results = data.get("query", {}).get("search", [])
        if not results:
            return f"維基百科找不到關於「{question}」的結果"

        output: list[str] = []
        for r in results[:2]:
            title = r.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", r.get("snippet", ""))
            output.append(f"【{title}】{snippet}")

        return "\n\n".join(output)
