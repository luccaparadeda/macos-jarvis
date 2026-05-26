import asyncio
import sys

from jarvis.audio import record_until_silence
from jarvis.brain import needs_vision, think_and_act
from jarvis.config import Settings
from jarvis.ears import transcribe
from jarvis.eyes import capture
from jarvis.hands import (
    discover_shortcuts, build_tool_schema,
    build_open_tool_schema, build_search_tool_schema, build_maintenance_tool_schema,
)
from jarvis.mouth import speak
from jarvis.wake import start_listener, stop_listener


async def pipeline_iteration(
    interrupt: asyncio.Event,
    tools: list[dict],
    conversation: list[dict],
    settings: Settings,
) -> None:
    audio_buf = await record_until_silence(interrupt, settings)
    if interrupt.is_set():
        return

    text = await transcribe(audio_buf, settings)
    if interrupt.is_set():
        return

    if not text.strip():
        await speak("I didn't catch that.", interrupt, settings)
        return

    print(f"[Jarvis] Heard: {text}")

    image = None
    if needs_vision(text, settings):
        try:
            image = await capture(settings)
            print("[Jarvis] Captured image.")
        except RuntimeError as e:
            print(f"[Jarvis] Camera error: {e}")
            await speak("I can't see right now.", interrupt, settings)
            if interrupt.is_set():
                return

    if interrupt.is_set():
        return

    try:
        response = await think_and_act(text, image, interrupt, tools, conversation, settings)
    except Exception as e:
        print(f"[Jarvis] Brain error: {e}")
        await speak("I couldn't reach my brain, try again.", interrupt, settings)
        return

    if interrupt.is_set():
        return

    if response:
        print(f"[Jarvis] Says: {response}")
        await speak(response, interrupt, settings)


async def main() -> None:
    settings = Settings()
    print("[Jarvis] Loading models and discovering shortcuts...")

    shortcut_names = await discover_shortcuts()
    tools = [
        build_tool_schema(shortcut_names),
        build_open_tool_schema(),
        build_search_tool_schema(),
        build_maintenance_tool_schema(),
    ] if shortcut_names else [
        build_open_tool_schema(),
        build_search_tool_schema(),
        build_maintenance_tool_schema(),
    ]
    print(f"[Jarvis] Found {len(shortcut_names)} shortcuts.")

    conversation: list[dict] = []
    wake_event = asyncio.Event()
    interrupt = asyncio.Event()
    loop = asyncio.get_event_loop()

    listener = await start_listener(wake_event, loop, settings.wake_model)
    print("[Jarvis] Listening for wake word... Say 'Hey Jarvis'!")

    try:
        while True:
            await wake_event.wait()
            wake_event.clear()
            interrupt.clear()
            print("[Jarvis] Wake word detected!")

            await pipeline_iteration(interrupt, tools, conversation, settings)
            print("[Jarvis] Ready for next command.")
    except KeyboardInterrupt:
        print("\n[Jarvis] Shutting down...")
    finally:
        await stop_listener(listener)


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
