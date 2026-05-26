# Jarvis: Streamlined macOS Spatial Assistant — Design Spec

**Date:** 2026-05-26
**Status:** Approved
**Platform:** macOS (Apple Silicon M3, 16GB Unified Memory)

## 1. Product Vision

A highly responsive, context-aware voice assistant embedded in the macOS ecosystem. Uses the iPhone as its eyes via Continuity Camera and Apple Shortcuts as its hands. Hybrid pipeline: local models for audio processing, cloud API (DeepSeek V4-Flash) for reasoning and tool routing. Architecture supports future migration to 100% local.

## 2. Architecture: Single-Process Async Event Loop

One Python process running an `asyncio` event loop. Each pipeline stage runs as a coroutine. Blocking ML inference (Whisper, Kokoro) runs in thread pool executors via `asyncio.run_in_executor`. OpenWakeWord runs in a dedicated background thread.

### Pipeline Flow

```
IDLE (OpenWakeWord listening)
  → Wake word "Jarvis" detected
  → Record audio until silence
  → Transcribe (Whisper via MLX)
  → Check if vision needed (keyword heuristic)
  → [If yes] Capture image from Continuity Camera
  → Send to DeepSeek V4-Flash (text + optional image + tool schema)
  → [Tool loop] Execute requested Apple Shortcuts, feed results back
  → Receive final text response
  → Synthesize speech (Kokoro via MLX-Audio)
  → Play audio
  → Resume IDLE
```

### Interrupt Support

OpenWakeWord runs continuously (never paused). If "Jarvis" is detected during an active pipeline run:

1. Set a shared `asyncio.Event` (`interrupt`).
2. Each pipeline stage checks `interrupt.is_set()` before proceeding.
3. TTS playback is stopped immediately via `sounddevice.stop()`.
4. In-flight API calls are cancelled.
5. Pipeline restarts with fresh recording.

## 3. Tech Stack

| Component | Technology | Execution |
|-----------|-----------|-----------|
| Wake Word | OpenWakeWord (`hey_jarvis` model) | Local (thread) |
| STT | mlx-whisper (whisper-tiny) | Local (MLX GPU) |
| Vision | OpenCV + Continuity Camera | Local |
| Brain/LLM | DeepSeek V4-Flash via OpenAI SDK | Cloud API |
| Shortcuts | `subprocess` → `shortcuts` CLI | Local |
| TTS | Kokoro via mlx-audio | Local (MLX GPU) |
| Audio I/O | sounddevice + numpy | Local |

## 4. Project Structure

```
local-jarvis/
├── pyproject.toml
├── .env                     # DEEPSEEK_API_KEY, model paths
├── src/
│   └── jarvis/
│       ├── __init__.py
│       ├── main.py          # Entry point, async loop, pipeline orchestration
│       ├── config.py         # Pydantic Settings from .env
│       ├── wake.py           # OpenWakeWord listener (background thread)
│       ├── ears.py           # Whisper STT
│       ├── eyes.py           # Continuity Camera capture
│       ├── brain.py          # DeepSeek client, tool loop, conversation
│       ├── hands.py          # Apple Shortcuts discovery + execution
│       ├── mouth.py          # Kokoro TTS
│       └── audio.py          # Mic capture, speaker playback
├── tests/
└── PRD.md
```

## 5. Dependencies

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "openwakeword",
    "mlx-whisper",
    "mlx-audio",
    "opencv-python",
    "openai",
    "sounddevice",
    "numpy",
    "pydantic-settings",
]
```

## 6. Module Interfaces

### config.py

```python
class Settings(BaseSettings):
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    whisper_model: str = "mlx-community/whisper-tiny"
    kokoro_model: str = "mlx-community/Kokoro-82M-bf16"
    wake_model: str = "hey_jarvis"
    camera_index: int = 0
    silence_threshold: float = 0.01
    silence_duration: float = 1.5
    vision_keywords: list[str] = ["look", "see", "show", "what is", "camera", "screen"]
```

### wake.py

```python
async def start_listener(wake_event: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None
async def stop_listener() -> None
```

Runs OpenWakeWord in a background thread. When wake word detected, sets `wake_event` from the thread via `loop.call_soon_threadsafe`. Runs continuously even during active pipeline (for interrupt support).

### audio.py

```python
async def record_until_silence(interrupt: asyncio.Event, settings: Settings) -> np.ndarray
```

Records from default mic using `sounddevice.InputStream`. Stops when silence exceeds `silence_duration` threshold or `interrupt` is set. Returns raw audio buffer as numpy array (16kHz, mono, float32).

### ears.py

```python
async def transcribe(audio: np.ndarray, settings: Settings) -> str
```

Runs `mlx_whisper.transcribe()` in a thread executor. Returns transcribed text.

### eyes.py

```python
async def capture(settings: Settings) -> str
```

Opens camera via `cv2.VideoCapture(camera_index)`, grabs a single frame, encodes as JPEG, returns base64 string. Releases camera immediately after capture.

### brain.py

```python
def needs_vision(text: str, settings: Settings) -> bool
async def think_and_act(
    text: str,
    image: str | None,
    interrupt: asyncio.Event,
    tools: list[dict],
    conversation: list[dict],
    settings: Settings,
) -> str
```

`needs_vision`: keyword scan against `settings.vision_keywords`.

`think_and_act`: Sends messages to DeepSeek via OpenAI SDK. If the response contains tool calls, executes them via `hands.run_shortcut()`, appends results, and re-calls the API. Loops until a final text response. Checks `interrupt` between iterations.

System prompt establishes Jarvis personality and describes available shortcuts.

### hands.py

```python
async def discover_shortcuts() -> list[dict]
async def run_shortcut(name: str, input_text: str | None = None) -> str
```

`discover_shortcuts`: Runs `shortcuts list`, parses output, generates a single OpenAI-compatible tool schema with `shortcut_name` as an enum.

`run_shortcut`: Executes `shortcuts run <name> [--input-text <text>]` via `asyncio.create_subprocess_exec`. Returns stdout on success, stderr on failure.

### mouth.py

```python
async def speak(text: str, interrupt: asyncio.Event, settings: Settings) -> None
```

Generates audio via Kokoro/mlx-audio in a thread executor. Plays via `sounddevice.play()`. Monitors `interrupt` during playback; calls `sounddevice.stop()` if set.

## 7. Dynamic Shortcut Discovery

At startup, `hands.discover_shortcuts()` queries all available shortcuts and generates a tool schema:

```json
{
  "type": "function",
  "function": {
    "name": "run_apple_shortcut",
    "description": "Execute a macOS Apple Shortcut by name.",
    "parameters": {
      "type": "object",
      "properties": {
        "shortcut_name": {
          "type": "string",
          "enum": ["<dynamically populated>"]
        },
        "input_text": {
          "type": "string",
          "description": "Optional text input to pass to the shortcut"
        }
      },
      "required": ["shortcut_name"]
    }
  }
}
```

Shortcut list refreshes every 10 minutes or on execution failure.

## 8. Conversation Context

- `brain.py` maintains a `list[dict]` of messages within the session.
- System prompt + last ~20 user/assistant/tool messages kept in context.
- Resets on process restart (no disk persistence for MVP).
- Tool call/result pairs included in history for coherent multi-step reasoning.

## 9. Error Handling

| Failure | Response |
|---------|----------|
| DeepSeek API timeout/error | Speak "I couldn't reach my brain, try again." Reset to idle. |
| Camera unavailable | Speak "I can't see right now." Proceed text-only. |
| Shortcut execution fails | Return stderr to DeepSeek for it to explain the failure. |
| Whisper produces empty text | Speak "I didn't catch that." Reset to idle. |
| Model loading fails at startup | Print error, exit. No recovery attempt. |

## 10. Performance Targets

| Metric | Target |
|--------|--------|
| Wake word to recording start | < 100ms |
| STT (whisper-tiny) | < 300ms |
| API roundtrip (text only) | < 1,000ms |
| API roundtrip (with image) | < 3,000ms |
| Shortcut execution initiation | < 200ms |
| Total memory (all models loaded) | < 4GB |

## 11. Future Milestones (Post-MVP)

- 100% local migration (Ollama with vision model, swap `base_url`)
- `launchd` daemonization (run on boot, no terminal)
- Persistent conversation history (SQLite)
- Voice activity detection improvements (replace silence threshold with energy-based VAD)
