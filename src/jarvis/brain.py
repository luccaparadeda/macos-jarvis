import asyncio
import json

import anthropic

from jarvis.config import Settings
from jarvis import hands

SYSTEM_PROMPT = (
    "You are Jarvis, a helpful and concise macOS voice assistant. "
    "You control the user's Mac through Apple Shortcuts and system tools. "
    "Keep responses short and conversational — they will be spoken aloud. "
    "When you execute a shortcut or tool, report the result naturally."
)

MAX_CONVERSATION_MESSAGES = 20

_client: anthropic.Anthropic | None = None


def _get_client(settings: Settings) -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def needs_vision(text: str, settings: Settings) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in settings.vision_keywords)


def _convert_tools_for_anthropic(tools: list[dict]) -> list[dict]:
    converted = []
    for tool in tools:
        fn = tool["function"]
        converted.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return converted


async def _execute_tool(name: str, args: dict) -> str:
    if name == "run_apple_shortcut":
        return await hands.run_shortcut(
            args["shortcut_name"], input_text=args.get("input_text")
        )
    elif name == "open_item":
        return await hands.open_item(
            args["path_or_app"], with_app=args.get("with_app")
        )
    elif name == "search_files":
        return await hands.search_files(args["query"])
    elif name == "system_maintenance":
        return await hands.system_maintenance(
            args["action"], dry_run=args.get("dry_run", True)
        )
    return f"Unknown tool: {name}"


async def think_and_act(
    text: str,
    image: str | None,
    interrupt: asyncio.Event,
    tools: list[dict],
    conversation: list[dict],
    settings: Settings,
) -> str:
    if interrupt.is_set():
        return ""

    client = _get_client(settings)

    if image:
        user_content = [
            {"type": "text", "text": text},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image,
                },
            },
        ]
    else:
        user_content = text

    conversation.append({"role": "user", "content": user_content})
    trimmed = conversation[-MAX_CONVERSATION_MESSAGES:]

    anthropic_tools = _convert_tools_for_anthropic(tools) if tools else []

    while not interrupt.is_set():
        kwargs = {
            "model": settings.anthropic_model,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": trimmed,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: client.messages.create(**kwargs)
        )

        if response.stop_reason != "tool_use":
            text_parts = [b.text for b in response.content if b.type == "text"]
            reply = " ".join(text_parts) if text_parts else ""
            conversation.append({"role": "assistant", "content": response.content})
            return reply

        conversation.append({"role": "assistant", "content": response.content})
        trimmed.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if interrupt.is_set():
                return ""

            result = await _execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        conversation.append({"role": "user", "content": tool_results})
        trimmed.append({"role": "user", "content": tool_results})

    return ""
