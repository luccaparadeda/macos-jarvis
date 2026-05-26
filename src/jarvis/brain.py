import asyncio
import json

from openai import OpenAI

from jarvis.config import Settings
from jarvis import hands

SYSTEM_PROMPT = (
    "You are Jarvis, a helpful and concise macOS voice assistant. "
    "You control the user's Mac through Apple Shortcuts and system tools. "
    "Keep responses short and conversational — they will be spoken aloud. "
    "When you execute a shortcut or tool, report the result naturally."
)

MAX_CONVERSATION_MESSAGES = 20

_client: OpenAI | None = None


def _get_client(settings: Settings) -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    return _client


def needs_vision(text: str, settings: Settings) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in settings.vision_keywords)


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

    if not conversation:
        conversation.append({"role": "system", "content": SYSTEM_PROMPT})

    if image:
        user_content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
        ]
    else:
        user_content = text

    conversation.append({"role": "user", "content": user_content})
    trimmed = [conversation[0]] + conversation[1:][-MAX_CONVERSATION_MESSAGES:]

    while not interrupt.is_set():
        kwargs = {"model": settings.deepseek_model, "messages": trimmed}
        if tools:
            kwargs["tools"] = tools

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: client.chat.completions.create(**kwargs)
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            conversation.append({"role": "assistant", "content": msg.content})
            return msg.content or ""

        assistant_msg = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }
        trimmed.append(assistant_msg)
        conversation.append(assistant_msg)

        for tc in msg.tool_calls:
            if interrupt.is_set():
                return ""
            args = json.loads(tc.function.arguments)
            result = await _execute_tool(tc.function.name, args)
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            trimmed.append(tool_msg)
            conversation.append(tool_msg)

    return ""
