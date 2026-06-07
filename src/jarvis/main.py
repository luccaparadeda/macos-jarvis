import asyncio
import sys
import time

from jarvis.audio import record_until_silence
from jarvis.brain import needs_vision, think_and_act
from jarvis.config import Settings
from jarvis.ears import transcribe
from jarvis.harness import build_context, build_harness_tool_schemas, init_harness, set_available_shortcuts
from jarvis.eyes import capture
from jarvis.hands import (
    discover_shortcuts, build_tool_schema,
    build_open_tool_schema, build_search_tool_schema, build_maintenance_tool_schema,
)
from jarvis.mouth import speak
from jarvis.wake import start_listener, stop_listener


def _log(stage: str, msg: str, start: float | None = None):
    elapsed = f" ({time.monotonic() - start:.1f}s)" if start else ""
    print(f"  [{stage}]{elapsed} {msg}")


async def pipeline_iteration(
    interrupt: asyncio.Event,
    tools: list[dict],
    conversation: list[dict],
    settings: Settings,
    listener=None,
    system_extra: str = "",
) -> None:
    t0 = time.monotonic()

    # 1. Record
    _log("Mic", "Pausing wake word, recording...")
    ambient = listener.ambient_level if listener else None
    if listener:
        listener.pause()
    audio_buf = await record_until_silence(interrupt, settings, ambient=ambient)
    if listener:
        listener.resume()
    if interrupt.is_set():
        return

    if audio_buf.size == 0:
        _log("Mic", "No voice captured", t0)
        await speak("I didn't catch that.", interrupt, settings)
        return

    duration_s = len(audio_buf) / 16000
    _log("Mic", f"Got {duration_s:.1f}s of audio", t0)

    # 2. Transcribe
    t1 = time.monotonic()
    _log("STT", "Transcribing...")
    text = await transcribe(audio_buf, settings)
    if interrupt.is_set():
        return

    if not text.strip():
        _log("STT", "Empty transcription, nothing heard", t1)
        await speak("I didn't catch that.", interrupt, settings)
        return

    _log("STT", f'"{text}"', t1)

    # 3. Vision check
    image = None
    if needs_vision(text, settings):
        t2 = time.monotonic()
        _log("Eyes", "Vision keywords detected, capturing camera...")
        try:
            image = await capture(settings)
            _log("Eyes", "Image captured", t2)
        except RuntimeError as e:
            _log("Eyes", f"Camera error: {e}", t2)
            await speak("I can't see right now.", interrupt, settings)
            if interrupt.is_set():
                return

    if interrupt.is_set():
        return

    # 4. Think
    t3 = time.monotonic()
    _log("Brain", f"Sending to Claude ({settings.anthropic_model})...")
    try:
        response = await think_and_act(
            text, image, interrupt, tools, conversation, settings,
            system_extra=system_extra,
        )
    except Exception as e:
        _log("Brain", f"ERROR: {e}", t3)
        await speak("I couldn't reach my brain, try again.", interrupt, settings)
        return

    if interrupt.is_set():
        return

    _log("Brain", f'"{response[:80]}{"..." if len(response) > 80 else ""}"', t3)

    # 5. Speak — pause the wake listener so Jarvis doesn't hear himself
    if response:
        t4 = time.monotonic()
        _log("TTS", "Generating speech...")
        if listener:
            listener.pause()
        await speak(response, interrupt, settings)
        if listener:
            listener.resume()
        _log("TTS", "Done speaking", t4)

    _log("Total", "Pipeline complete", t0)


async def main() -> None:
    settings = Settings()
    print("[Jarvis] Loading models and discovering shortcuts...")

    shortcut_names = await discover_shortcuts()
    home = init_harness()
    set_available_shortcuts(shortcut_names)
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
    tools.extend(build_harness_tool_schemas())
    print(f"[Jarvis] Found {len(shortcut_names)} shortcuts: {', '.join(shortcut_names)}")
    print(f"[Jarvis] Brain: {settings.anthropic_model}")
    print(f"[Jarvis] Harness ready at {home}")

    conversation: list[dict] = []
    system_extra = build_context()
    wake_event = asyncio.Event()
    interrupt = asyncio.Event()
    loop = asyncio.get_running_loop()

    listener = await start_listener(wake_event, loop, settings.wake_model, settings.wake_threshold)
    print("[Jarvis] Listening for wake word... Say 'Hey Jarvis'!")
    print()

    try:
        while True:
            await wake_event.wait()
            wake_event.clear()
            interrupt.clear()
            print(">>> Wake word detected!")

            await pipeline_iteration(interrupt, tools, conversation, settings, listener, system_extra=system_extra)
            wake_event.clear()  # drop any wake triggers that fired mid-pipeline
            print()
    except KeyboardInterrupt:
        print("\n[Jarvis] Shutting down...")
    finally:
        await stop_listener(listener)


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
