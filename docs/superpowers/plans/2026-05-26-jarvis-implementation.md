# Jarvis macOS Spatial Assistant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a voice-activated macOS assistant that listens for "Jarvis", transcribes speech, optionally captures camera images, routes to DeepSeek V4-Flash for reasoning + Apple Shortcut tool calls, and speaks the response back.

**Architecture:** Single-process Python asyncio event loop. Blocking ML inference (Whisper STT, Kokoro TTS) runs in thread pool executors. OpenWakeWord runs in a dedicated background thread, always active for interrupt support. DeepSeek V4-Flash via OpenAI SDK handles reasoning and tool routing.

**Tech Stack:** Python 3.11+, uv, asyncio, OpenWakeWord, mlx-whisper, mlx-audio (Kokoro), OpenCV, OpenAI SDK, sounddevice, numpy, pydantic-settings

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, entry point |
| `.env.example` | Template for required env vars |
| `.gitignore` | Standard Python + .env ignores |
| `src/jarvis/__init__.py` | Package marker |
| `src/jarvis/config.py` | Pydantic Settings model loaded from `.env` |
| `src/jarvis/hands.py` | Dynamic shortcut discovery + subprocess execution |
| `src/jarvis/audio.py` | Mic recording with silence detection + interrupt |
| `src/jarvis/ears.py` | Whisper STT via mlx-whisper |
| `src/jarvis/eyes.py` | Continuity Camera capture via OpenCV |
| `src/jarvis/brain.py` | DeepSeek client, vision check, tool loop, conversation |
| `src/jarvis/mouth.py` | Kokoro TTS via mlx-audio |
| `src/jarvis/wake.py` | OpenWakeWord background thread listener |
| `src/jarvis/main.py` | Entry point, async pipeline orchestration, interrupt handling |
| `tests/conftest.py` | Shared fixtures |
| `tests/test_config.py` | Config loading tests |
| `tests/test_hands.py` | Shortcut discovery + execution tests |
| `tests/test_brain.py` | Vision check + tool loop tests |
| `tests/test_audio.py` | Silence detection logic tests |
| `tests/test_eyes.py` | Camera capture tests |
| `tests/test_main.py` | Pipeline integration tests |

---

### Task 1: Project Scaffold & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/jarvis/__init__.py`

- [ ] **Step 1: Initialize uv project with Python 3.11**

```bash
cd /Users/luccaparadeda/Programming/local-jarvis
uv init --python 3.11 --no-readme
```

This creates `pyproject.toml` and `.python-version`. If uv creates a `hello.py` or `main.py` at the root, delete it.

- [ ] **Step 2: Edit pyproject.toml to match our spec**

Replace the contents of `pyproject.toml` with:

```toml
[project]
name = "jarvis"
version = "0.1.0"
description = "Streamlined macOS Spatial Voice Assistant"
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
    "pyaudio",
]

[project.scripts]
jarvis = "jarvis.main:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"

[tool.hatch.build.targets.wheel]
packages = ["src/jarvis"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[dependency-groups]
dev = [
    "pytest",
    "pytest-asyncio",
]
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.env
.venv/
dist/
*.egg-info/
.python-version
```

- [ ] **Step 4: Create .env.example**

```
DEEPSEEK_API_KEY=your-key-here
```

- [ ] **Step 5: Create src/jarvis/__init__.py**

```bash
mkdir -p src/jarvis tests
```

```python
# src/jarvis/__init__.py
```

(Empty file — package marker only.)

- [ ] **Step 6: Install dependencies**

```bash
uv sync
```

This creates the venv and installs all dependencies. Verify it succeeds without errors.

- [ ] **Step 7: Verify pytest runs**

```bash
uv run pytest --co
```

Expected: "no tests ran" (no test files yet), but pytest itself loads without errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/jarvis/__init__.py uv.lock
git commit -m "feat: scaffold project with uv, dependencies, and package structure"
```

---

### Task 2: Configuration Module

**Files:**
- Create: `src/jarvis/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import os
from jarvis.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
    settings = Settings()
    assert settings.deepseek_api_key == "test-key-123"
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-chat"


def test_settings_defaults():
    settings = Settings(deepseek_api_key="k")
    assert settings.whisper_model == "mlx-community/whisper-tiny"
    assert settings.kokoro_model == "mlx-community/Kokoro-82M-bf16"
    assert settings.wake_model == "hey_jarvis"
    assert settings.camera_index == 0
    assert settings.silence_threshold == 0.01
    assert settings.silence_duration == 1.5
    assert "look" in settings.vision_keywords
    assert "see" in settings.vision_keywords


def test_settings_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("CAMERA_INDEX", "2")
    settings = Settings()
    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.camera_index == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.config'`

- [ ] **Step 3: Write the implementation**

Create `src/jarvis/config.py`:

```python
from pydantic_settings import BaseSettings


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
    vision_keywords: list[str] = [
        "look",
        "see",
        "show",
        "what is",
        "camera",
        "screen",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/config.py tests/test_config.py
git commit -m "feat: add Settings config module with env loading"
```

---

### Task 3: Apple Shortcuts — Discovery & Execution

**Files:**
- Create: `src/jarvis/hands.py`
- Create: `tests/test_hands.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hands.py`:

```python
import json
import pytest
from unittest.mock import AsyncMock, patch

from jarvis.hands import discover_shortcuts, run_shortcut, build_tool_schema


@pytest.mark.asyncio
async def test_discover_shortcuts_parses_output():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (
        b"Add new event\nWhat's on today?\nTake a Break\n",
        b"",
    )
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        names = await discover_shortcuts()

    assert names == ["Add new event", "What's on today?", "Take a Break"]


@pytest.mark.asyncio
async def test_discover_shortcuts_skips_empty_lines():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (
        b"Shortcut A\n\nShortcut B\n\n",
        b"",
    )
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        names = await discover_shortcuts()

    assert names == ["Shortcut A", "Shortcut B"]


def test_build_tool_schema():
    names = ["Add new event", "Take a Break"]
    schema = build_tool_schema(names)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "run_apple_shortcut"
    params = schema["function"]["parameters"]
    assert params["properties"]["shortcut_name"]["enum"] == names
    assert "input_text" in params["properties"]
    assert params["required"] == ["shortcut_name"]


@pytest.mark.asyncio
async def test_run_shortcut_success():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"Event created", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await run_shortcut("Add new event", input_text="Meeting at 3pm")

    assert result == "Event created"
    mock_exec.assert_called_once_with(
        "shortcuts", "run", "Add new event", "--input-text", "Meeting at 3pm",
        stdout=-1, stderr=-1,
    )


@pytest.mark.asyncio
async def test_run_shortcut_no_input():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"Done", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await run_shortcut("Take a Break")

    assert result == "Done"
    mock_exec.assert_called_once_with(
        "shortcuts", "run", "Take a Break",
        stdout=-1, stderr=-1,
    )


@pytest.mark.asyncio
async def test_run_shortcut_failure_returns_stderr():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"Shortcut not found")
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await run_shortcut("Nonexistent")

    assert result == "Error: Shortcut not found"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_hands.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.hands'`

- [ ] **Step 3: Write the implementation**

Create `src/jarvis/hands.py`:

```python
import asyncio
import subprocess


async def discover_shortcuts() -> list[str]:
    proc = await asyncio.create_subprocess_exec(
        "shortcuts", "list",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    lines = stdout.decode().strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_tool_schema(shortcut_names: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "run_apple_shortcut",
            "description": (
                "Execute a macOS Apple Shortcut by name. "
                "Use this to control apps, send messages, manage calendar, "
                "take notes, and automate the Mac."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "shortcut_name": {
                        "type": "string",
                        "enum": shortcut_names,
                    },
                    "input_text": {
                        "type": "string",
                        "description": "Optional text input to pass to the shortcut",
                    },
                },
                "required": ["shortcut_name"],
            },
        },
    }


async def run_shortcut(name: str, input_text: str | None = None) -> str:
    cmd = ["shortcuts", "run", name]
    if input_text is not None:
        cmd.extend(["--input-text", input_text])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return f"Error: {stderr.decode().strip()}"
    return stdout.decode().strip()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_hands.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/hands.py tests/test_hands.py
git commit -m "feat: add Apple Shortcuts discovery and execution"
```

---

### Task 3b: System Tools — open, mdfind, Mole

**Files:**
- Modify: `src/jarvis/hands.py`
- Modify: `tests/test_hands.py`

Extends hands.py with three new system tool functions and their tool schemas.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hands.py`:

```python
from jarvis.hands import (
    open_item, search_files, system_maintenance,
    build_open_tool_schema, build_search_tool_schema, build_maintenance_tool_schema,
)


@pytest.mark.asyncio
async def test_open_item_app():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await open_item("Spotify")

    assert "opened" in result.lower() or result == ""
    mock_exec.assert_called_once_with(
        "open", "-a", "Spotify",
        stdout=-1, stderr=-1,
    )


@pytest.mark.asyncio
async def test_open_item_file_with_app():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await open_item("/Users/test/doc.numbers", with_app="Numbers")

    mock_exec.assert_called_once_with(
        "open", "-a", "Numbers", "/Users/test/doc.numbers",
        stdout=-1, stderr=-1,
    )


@pytest.mark.asyncio
async def test_open_item_failure():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"The file does not exist.")
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await open_item("/nonexistent/file.txt")

    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_search_files():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (
        b"/Users/test/invoice.pdf\n/Users/test/Downloads/invoice2.pdf\n",
        b"",
    )
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await search_files("name:invoice.pdf")

    assert "/Users/test/invoice.pdf" in result
    assert "/Users/test/Downloads/invoice2.pdf" in result


@pytest.mark.asyncio
async def test_search_files_no_results():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await search_files("name:nonexistent_file_xyz.pdf")

    assert "no files found" in result.lower()


@pytest.mark.asyncio
async def test_system_maintenance_dry_run():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"Would remove 2.3GB of cache files", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await system_maintenance("clean", dry_run=True)

    assert "2.3GB" in result
    args = mock_exec.call_args[0]
    assert "--dry-run" in args


@pytest.mark.asyncio
async def test_system_maintenance_status():
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"CPU: 12% | Memory: 8.2GB/16GB", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await system_maintenance("status")

    assert "CPU" in result
    args = mock_exec.call_args[0]
    assert "--dry-run" not in args  # status is non-destructive


def test_build_open_tool_schema():
    schema = build_open_tool_schema()
    assert schema["function"]["name"] == "open_item"
    assert "path_or_app" in schema["function"]["parameters"]["properties"]


def test_build_search_tool_schema():
    schema = build_search_tool_schema()
    assert schema["function"]["name"] == "search_files"
    assert "query" in schema["function"]["parameters"]["properties"]


def test_build_maintenance_tool_schema():
    schema = build_maintenance_tool_schema()
    assert schema["function"]["name"] == "system_maintenance"
    props = schema["function"]["parameters"]["properties"]
    assert "action" in props
    assert "dry_run" in props
    assert props["action"]["enum"] == ["clean", "analyze", "status", "purge", "optimize"]
```

- [ ] **Step 2: Run test to verify new tests fail**

```bash
uv run pytest tests/test_hands.py -v
```

Expected: New tests fail with `ImportError` (functions not defined yet). Old tests still pass.

- [ ] **Step 3: Add implementations to hands.py**

Append to `src/jarvis/hands.py`:

```python
DESTRUCTIVE_ACTIONS = {"clean", "purge", "optimize"}
NON_DESTRUCTIVE_ACTIONS = {"analyze", "status"}


def build_open_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "open_item",
            "description": (
                "Open a file, folder, or application on macOS. "
                "Like double-clicking in Finder."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path_or_app": {
                        "type": "string",
                        "description": "File path, folder path, or application name to open",
                    },
                    "with_app": {
                        "type": "string",
                        "description": "Optional: open the file with a specific application",
                    },
                },
                "required": ["path_or_app"],
            },
        },
    }


def build_search_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search for files on macOS using Spotlight (mdfind). "
                "Searches file names and contents instantly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query — file name, content keywords, or metadata filter "
                            "(e.g. 'name:invoice.pdf', 'kMDItemContentType=com.adobe.pdf')"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }


def build_maintenance_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "system_maintenance",
            "description": (
                "Run Mac system maintenance using Mole (mo). "
                "Clean caches, analyze disk usage, check system status, or purge build artifacts. "
                "Destructive commands default to dry-run mode for safety."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["clean", "analyze", "status", "purge", "optimize"],
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "If true (default), show what would be done without doing it. "
                            "Set false only with explicit user confirmation."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    }


async def open_item(path_or_app: str, with_app: str | None = None) -> str:
    if with_app:
        cmd = ["open", "-a", with_app, path_or_app]
    else:
        cmd = ["open", "-a", path_or_app]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return f"Error: {stderr.decode().strip()}"
    return f"Opened {path_or_app}"


async def search_files(query: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "mdfind", query,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    output = stdout.decode().strip()
    if not output:
        return "No files found."

    lines = output.splitlines()[:20]
    return "\n".join(lines)


async def system_maintenance(action: str, dry_run: bool = True) -> str:
    cmd = ["mo", action]
    if action in DESTRUCTIVE_ACTIONS and dry_run:
        cmd.append("--dry-run")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return f"Error: {stderr.decode().strip()}"
    return stdout.decode().strip()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_hands.py -v
```

Expected: All tests pass (original 6 + new 10 = 16 total).

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/hands.py tests/test_hands.py
git commit -m "feat: add system tools — open, mdfind, and Mole maintenance"
```

---

### Task 4: Audio Recording with Silence Detection

**Files:**
- Create: `src/jarvis/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio.py`:

```python
import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from jarvis.audio import record_until_silence
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"deepseek_api_key": "k", "silence_threshold": 0.01, "silence_duration": 0.1}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_record_stops_on_silence():
    """Simulate: one loud chunk, then silence. Should return the loud chunk."""
    settings = _make_settings(silence_threshold=0.5, silence_duration=0.05)
    interrupt = asyncio.Event()

    loud_chunk = np.ones(1600, dtype=np.float32)
    silent_chunk = np.zeros(1600, dtype=np.float32)
    chunks = [loud_chunk, silent_chunk, silent_chunk, silent_chunk]
    call_count = 0

    def fake_input_stream(**kwargs):
        stream = MagicMock()
        callback = kwargs["callback"]

        def start():
            nonlocal call_count
            for chunk in chunks:
                callback(chunk.reshape(-1, 1), None, None, None)
                call_count += 1

        stream.start = start
        stream.stop = MagicMock()
        stream.close = MagicMock()
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)
        return stream

    with patch("sounddevice.InputStream", side_effect=fake_input_stream):
        result = await record_until_silence(interrupt, settings)

    assert isinstance(result, np.ndarray)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_record_stops_on_interrupt():
    settings = _make_settings()
    interrupt = asyncio.Event()
    interrupt.set()

    with patch("sounddevice.InputStream") as mock_cls:
        stream = MagicMock()
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)
        mock_cls.return_value = stream

        result = await record_until_silence(interrupt, settings)

    assert isinstance(result, np.ndarray)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_audio.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.audio'`

- [ ] **Step 3: Write the implementation**

Create `src/jarvis/audio.py`:

```python
import asyncio
import time

import numpy as np
import sounddevice as sd

from jarvis.config import Settings

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600  # 100ms at 16kHz


async def record_until_silence(interrupt: asyncio.Event, settings: Settings) -> np.ndarray:
    loop = asyncio.get_event_loop()
    chunks: list[np.ndarray] = []
    last_voice_time = time.monotonic()
    recording_done = asyncio.Event()

    def callback(indata, frames, time_info, status):
        nonlocal last_voice_time
        if interrupt.is_set():
            recording_done._loop = loop
            loop.call_soon_threadsafe(recording_done.set)
            return

        chunk = indata[:, 0].copy()
        chunks.append(chunk)

        amplitude = np.abs(chunk).mean()
        if amplitude > settings.silence_threshold:
            last_voice_time = time.monotonic()
        elif time.monotonic() - last_voice_time > settings.silence_duration:
            loop.call_soon_threadsafe(recording_done.set)

    if interrupt.is_set():
        return np.array([], dtype=np.float32)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=callback,
    ):
        await recording_done.wait()

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_audio.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/audio.py tests/test_audio.py
git commit -m "feat: add mic recording with silence detection and interrupt"
```

---

### Task 5: Speech-to-Text (Ears)

**Files:**
- Create: `src/jarvis/ears.py`
- Create: `tests/test_ears.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ears.py`:

```python
import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from jarvis.ears import transcribe
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"deepseek_api_key": "k"}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_transcribe_returns_text():
    settings = _make_settings(whisper_model="mlx-community/whisper-tiny")
    audio = np.random.randn(16000).astype(np.float32)

    with patch("jarvis.ears.mlx_whisper") as mock_whisper:
        mock_whisper.transcribe.return_value = {"text": " Hello Jarvis "}
        result = await transcribe(audio, settings)

    assert result == "Hello Jarvis"
    mock_whisper.transcribe.assert_called_once()
    call_args = mock_whisper.transcribe.call_args
    assert call_args[1]["path_or_hf_repo"] == "mlx-community/whisper-tiny"


@pytest.mark.asyncio
async def test_transcribe_empty_audio():
    settings = _make_settings()
    audio = np.array([], dtype=np.float32)

    with patch("jarvis.ears.mlx_whisper") as mock_whisper:
        mock_whisper.transcribe.return_value = {"text": ""}
        result = await transcribe(audio, settings)

    assert result == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_ears.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.ears'`

- [ ] **Step 3: Write the implementation**

Create `src/jarvis/ears.py`:

```python
import asyncio
import functools

import mlx_whisper
import numpy as np

from jarvis.config import Settings


async def transcribe(audio: np.ndarray, settings: Settings) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        functools.partial(
            mlx_whisper.transcribe,
            audio,
            path_or_hf_repo=settings.whisper_model,
        ),
    )
    return result["text"].strip()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_ears.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/ears.py tests/test_ears.py
git commit -m "feat: add Whisper STT module"
```

---

### Task 6: Camera Capture (Eyes)

**Files:**
- Create: `src/jarvis/eyes.py`
- Create: `tests/test_eyes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eyes.py`:

```python
import base64
import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from jarvis.eyes import capture
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"deepseek_api_key": "k", "camera_index": 0}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_capture_returns_base64_jpeg():
    settings = _make_settings()
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, fake_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc:
        with patch("cv2.imencode") as mock_enc:
            _, real_buf = __import__("cv2").imencode(".jpg", fake_frame)
            mock_enc.return_value = (True, real_buf)

            result = await capture(settings)

    mock_vc.assert_called_once_with(0)
    mock_cap.release.assert_called_once()

    decoded = base64.b64decode(result)
    assert len(decoded) > 0


@pytest.mark.asyncio
async def test_capture_raises_on_camera_failure():
    settings = _make_settings()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(RuntimeError, match="Cannot open camera"):
            await capture(settings)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_eyes.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.eyes'`

- [ ] **Step 3: Write the implementation**

Create `src/jarvis/eyes.py`:

```python
import asyncio
import base64
import functools

import cv2
import numpy as np

from jarvis.config import Settings


def _capture_sync(camera_index: int) -> str:
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        success, buf = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")

        return base64.b64encode(buf.tobytes()).decode("utf-8")
    finally:
        cap.release()


async def capture(settings: Settings) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(_capture_sync, settings.camera_index),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_eyes.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/eyes.py tests/test_eyes.py
git commit -m "feat: add Continuity Camera capture module"
```

---

### Task 7: Brain — Vision Check & DeepSeek Tool Loop

**Files:**
- Create: `src/jarvis/brain.py`
- Create: `tests/test_brain.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_brain.py`:

```python
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jarvis.brain import needs_vision, think_and_act
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"deepseek_api_key": "test-key"}
    defaults.update(kwargs)
    return Settings(**defaults)


class TestNeedsVision:
    def test_look_keyword(self):
        s = _make_settings()
        assert needs_vision("look at this document", s) is True

    def test_see_keyword(self):
        s = _make_settings()
        assert needs_vision("can you see what's on my desk", s) is True

    def test_what_is_keyword(self):
        s = _make_settings()
        assert needs_vision("what is this thing", s) is True

    def test_no_vision_keyword(self):
        s = _make_settings()
        assert needs_vision("set a timer for 5 minutes", s) is False

    def test_case_insensitive(self):
        s = _make_settings()
        assert needs_vision("LOOK at the Screen", s) is True


class TestThinkAndAct:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = [{"type": "function", "function": {"name": "run_apple_shortcut"}}]

        mock_message = MagicMock()
        mock_message.content = "Hello! How can I help?"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await think_and_act("hello", None, interrupt, tools, conversation, settings)

        assert result == "Hello! How can I help?"
        assert len(conversation) == 3  # system + user + assistant

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = [{"type": "function", "function": {"name": "run_apple_shortcut"}}]

        # First response: tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "run_apple_shortcut"
        mock_tool_call.function.arguments = json.dumps({
            "shortcut_name": "What's on today?",
        })

        mock_msg1 = MagicMock()
        mock_msg1.content = None
        mock_msg1.tool_calls = [mock_tool_call]

        mock_choice1 = MagicMock()
        mock_choice1.message = mock_msg1

        mock_resp1 = MagicMock()
        mock_resp1.choices = [mock_choice1]

        # Second response: text
        mock_msg2 = MagicMock()
        mock_msg2.content = "You have 3 meetings today."
        mock_msg2.tool_calls = None

        mock_choice2 = MagicMock()
        mock_choice2.message = mock_msg2

        mock_resp2 = MagicMock()
        mock_resp2.choices = [mock_choice2]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(
                side_effect=[mock_resp1, mock_resp2]
            )
            mock_get_client.return_value = mock_client

            with patch("jarvis.hands.run_shortcut", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = "Meeting 1, Meeting 2, Meeting 3"
                result = await think_and_act(
                    "what's on my calendar", None, interrupt, tools, conversation, settings,
                )

        assert result == "You have 3 meetings today."
        mock_run.assert_called_once_with("What's on today?", input_text=None)

    @pytest.mark.asyncio
    async def test_interrupt_stops_tool_loop(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = []

        # Set interrupt before calling
        interrupt.set()

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            result = await think_and_act("hello", None, interrupt, tools, conversation, settings)

        assert result == ""
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_image_included_in_message(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = []

        mock_message = MagicMock()
        mock_message.content = "I see a laptop on the desk."
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await think_and_act(
                "look at my desk", "base64imgdata", interrupt, tools, conversation, settings,
            )

        assert result == "I see a laptop on the desk."
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_msg = messages[-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][1]["type"] == "image_url"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_brain.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.brain'`

- [ ] **Step 3: Write the implementation**

Create `src/jarvis/brain.py`:

```python
import asyncio
import json

from openai import OpenAI

from jarvis.config import Settings
from jarvis import hands

SYSTEM_PROMPT = (
    "You are Jarvis, a helpful and concise macOS voice assistant. "
    "You control the user's Mac through Apple Shortcuts. "
    "Keep responses short and conversational — they will be spoken aloud. "
    "When you execute a shortcut, report the result naturally."
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
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            },
        ]
    else:
        user_content = text

    conversation.append({"role": "user", "content": user_content})

    trimmed = [conversation[0]] + conversation[1:][-MAX_CONVERSATION_MESSAGES:]

    while not interrupt.is_set():
        kwargs = {
            "model": settings.deepseek_model,
            "messages": trimmed,
        }
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

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
            trimmed.append(tool_msg)
            conversation.append(tool_msg)

    return ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_brain.py -v
```

Expected: 7 passed (4 needs_vision + 4 think_and_act tests... actually 5 needs_vision + 4 think_and_act)

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/brain.py tests/test_brain.py
git commit -m "feat: add DeepSeek brain with vision check and tool loop"
```

---

### Task 8: Text-to-Speech (Mouth)

**Files:**
- Create: `src/jarvis/mouth.py`
- Create: `tests/test_mouth.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mouth.py`:

```python
import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from jarvis.mouth import speak
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"deepseek_api_key": "k"}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_speak_generates_and_plays_audio():
    settings = _make_settings()
    interrupt = asyncio.Event()
    fake_audio = np.random.randn(16000).astype(np.float32)

    with patch("jarvis.mouth._generate_audio", return_value=(fake_audio, 24000)) as mock_gen:
        with patch("sounddevice.play") as mock_play:
            with patch("sounddevice.wait") as mock_wait:
                await speak("Hello there", interrupt, settings)

    mock_gen.assert_called_once()
    mock_play.assert_called_once()
    call_args = mock_play.call_args
    assert call_args[1]["samplerate"] == 24000


@pytest.mark.asyncio
async def test_speak_stops_on_interrupt():
    settings = _make_settings()
    interrupt = asyncio.Event()
    interrupt.set()

    with patch("jarvis.mouth._generate_audio") as mock_gen:
        with patch("sounddevice.play") as mock_play:
            await speak("Hello", interrupt, settings)

    mock_gen.assert_not_called()
    mock_play.assert_not_called()


@pytest.mark.asyncio
async def test_speak_empty_text_does_nothing():
    settings = _make_settings()
    interrupt = asyncio.Event()

    with patch("jarvis.mouth._generate_audio") as mock_gen:
        await speak("", interrupt, settings)

    mock_gen.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_mouth.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.mouth'`

- [ ] **Step 3: Write the implementation**

Create `src/jarvis/mouth.py`:

```python
import asyncio
import functools

import numpy as np
import sounddevice as sd

from jarvis.config import Settings

_tts_pipeline = None


def _get_tts_pipeline(settings: Settings):
    global _tts_pipeline
    if _tts_pipeline is None:
        from mlx_audio.tts import TTS
        _tts_pipeline = TTS("mlx-community/Kokoro-82M-bf16")
    return _tts_pipeline


def _generate_audio(text: str, settings: Settings) -> tuple[np.ndarray, int]:
    tts = _get_tts_pipeline(settings)
    result = tts.generate(text=text)
    audio = result["audio"]
    sample_rate = result["sample_rate"]
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    return audio, sample_rate


async def speak(text: str, interrupt: asyncio.Event, settings: Settings) -> None:
    if not text or interrupt.is_set():
        return

    loop = asyncio.get_event_loop()
    audio, sample_rate = await loop.run_in_executor(
        None,
        functools.partial(_generate_audio, text, settings),
    )

    if interrupt.is_set():
        return

    sd.play(audio, samplerate=sample_rate)

    duration = len(audio) / sample_rate
    poll_interval = 0.1
    elapsed = 0.0
    while elapsed < duration:
        if interrupt.is_set():
            sd.stop()
            return
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    sd.wait()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_mouth.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/mouth.py tests/test_mouth.py
git commit -m "feat: add Kokoro TTS speech synthesis module"
```

---

### Task 9: Wake Word Listener

**Files:**
- Create: `src/jarvis/wake.py`
- Create: `tests/test_wake.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_wake.py`:

```python
import asyncio
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from jarvis.wake import WakeWordListener


@pytest.mark.asyncio
async def test_listener_sets_event_on_detection():
    loop = asyncio.get_event_loop()
    wake_event = asyncio.Event()

    mock_oww = MagicMock()
    # Simulate: first call detects wake word, second call raises to stop
    prediction = {"hey_jarvis": 0.9}
    mock_oww.predict.side_effect = [prediction, KeyboardInterrupt]

    mock_stream = MagicMock()
    mock_stream.read.return_value = b"\x00" * 2560

    with patch("jarvis.wake.Model", return_value=mock_oww):
        with patch("jarvis.wake.PyAudio") as mock_pyaudio_cls:
            mock_pa = MagicMock()
            mock_pa.open.return_value = mock_stream
            mock_pa.get_format_from_width.return_value = 8  # paInt16
            mock_pyaudio_cls.return_value = mock_pa

            listener = WakeWordListener(wake_event, loop, threshold=0.5)

            # Run in thread briefly
            import threading
            t = threading.Thread(target=listener._listen_loop, daemon=True)
            t.start()
            t.join(timeout=1.0)

    assert wake_event.is_set()


def test_listener_ignores_low_confidence():
    loop = asyncio.new_event_loop()
    wake_event = asyncio.Event()

    mock_oww = MagicMock()
    prediction = {"hey_jarvis": 0.2}
    mock_oww.predict.side_effect = [prediction, KeyboardInterrupt]

    mock_stream = MagicMock()
    mock_stream.read.return_value = b"\x00" * 2560

    with patch("jarvis.wake.Model", return_value=mock_oww):
        with patch("jarvis.wake.PyAudio") as mock_pyaudio_cls:
            mock_pa = MagicMock()
            mock_pa.open.return_value = mock_stream
            mock_pa.get_format_from_width.return_value = 8
            mock_pyaudio_cls.return_value = mock_pa

            listener = WakeWordListener(wake_event, loop, threshold=0.5)

            import threading
            t = threading.Thread(target=listener._listen_loop, daemon=True)
            t.start()
            t.join(timeout=1.0)

    assert not wake_event.is_set()
    loop.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_wake.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.wake'`

- [ ] **Step 3: Write the implementation**

Create `src/jarvis/wake.py`:

```python
import asyncio
import threading

from openwakeword import Model
from pyaudio import PyAudio

CHUNK_SIZE = 1280  # ~80ms at 16kHz
SAMPLE_RATE = 16000
FORMAT_WIDTH = 2  # 16-bit


class WakeWordListener:
    def __init__(
        self,
        wake_event: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
    ):
        self._wake_event = wake_event
        self._loop = loop
        self._model_name = model_name
        self._threshold = threshold
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _listen_loop(self) -> None:
        model = Model(wakeword_models=[self._model_name])
        pa = PyAudio()
        stream = pa.open(
            rate=SAMPLE_RATE,
            channels=1,
            format=pa.get_format_from_width(FORMAT_WIDTH),
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        try:
            while self._running:
                audio_data = stream.read(CHUNK_SIZE)
                predictions = model.predict(audio_data)
                for key, score in predictions.items():
                    if score > self._threshold:
                        self._loop.call_soon_threadsafe(self._wake_event.set)
        except (KeyboardInterrupt, OSError):
            pass
        finally:
            stream.close()
            pa.terminate()


async def start_listener(
    wake_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
    model_name: str = "hey_jarvis",
    threshold: float = 0.5,
) -> WakeWordListener:
    listener = WakeWordListener(wake_event, loop, model_name, threshold)
    listener.start()
    return listener


async def stop_listener(listener: WakeWordListener) -> None:
    listener.stop()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_wake.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/wake.py tests/test_wake.py
git commit -m "feat: add OpenWakeWord background listener"
```

---

### Task 10: Main Pipeline Orchestration

**Files:**
- Create: `src/jarvis/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_main.py`:

```python
import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jarvis.main import pipeline_iteration
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"deepseek_api_key": "test-key"}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_pipeline_text_only():
    """Full pipeline: record -> transcribe -> think (no vision) -> speak."""
    settings = _make_settings()
    interrupt = asyncio.Event()
    conversation: list[dict] = []
    tools = [{"type": "function", "function": {"name": "run_apple_shortcut"}}]
    audio_buf = np.random.randn(16000).astype(np.float32)

    with patch("jarvis.main.record_until_silence", new_callable=AsyncMock, return_value=audio_buf):
        with patch("jarvis.main.transcribe", new_callable=AsyncMock, return_value="set a timer"):
            with patch("jarvis.main.needs_vision", return_value=False):
                with patch("jarvis.main.think_and_act", new_callable=AsyncMock, return_value="Timer set!"):
                    with patch("jarvis.main.speak", new_callable=AsyncMock) as mock_speak:
                        await pipeline_iteration(interrupt, tools, conversation, settings)

    mock_speak.assert_called_once_with("Timer set!", interrupt, settings)


@pytest.mark.asyncio
async def test_pipeline_with_vision():
    """Full pipeline with vision: record -> transcribe -> capture -> think -> speak."""
    settings = _make_settings()
    interrupt = asyncio.Event()
    conversation: list[dict] = []
    tools = []
    audio_buf = np.random.randn(16000).astype(np.float32)

    with patch("jarvis.main.record_until_silence", new_callable=AsyncMock, return_value=audio_buf):
        with patch("jarvis.main.transcribe", new_callable=AsyncMock, return_value="look at my desk"):
            with patch("jarvis.main.needs_vision", return_value=True):
                with patch("jarvis.main.capture", new_callable=AsyncMock, return_value="base64img"):
                    with patch("jarvis.main.think_and_act", new_callable=AsyncMock, return_value="I see a laptop.") as mock_think:
                        with patch("jarvis.main.speak", new_callable=AsyncMock):
                            await pipeline_iteration(interrupt, tools, conversation, settings)

    mock_think.assert_called_once()
    call_args = mock_think.call_args
    assert call_args[0][1] == "base64img"  # image argument


@pytest.mark.asyncio
async def test_pipeline_empty_transcription_speaks_error():
    settings = _make_settings()
    interrupt = asyncio.Event()
    conversation: list[dict] = []
    tools = []
    audio_buf = np.random.randn(16000).astype(np.float32)

    with patch("jarvis.main.record_until_silence", new_callable=AsyncMock, return_value=audio_buf):
        with patch("jarvis.main.transcribe", new_callable=AsyncMock, return_value=""):
            with patch("jarvis.main.speak", new_callable=AsyncMock) as mock_speak:
                await pipeline_iteration(interrupt, tools, conversation, settings)

    mock_speak.assert_called_once_with("I didn't catch that.", interrupt, settings)


@pytest.mark.asyncio
async def test_pipeline_camera_failure_falls_back_to_text():
    settings = _make_settings()
    interrupt = asyncio.Event()
    conversation: list[dict] = []
    tools = []
    audio_buf = np.random.randn(16000).astype(np.float32)

    with patch("jarvis.main.record_until_silence", new_callable=AsyncMock, return_value=audio_buf):
        with patch("jarvis.main.transcribe", new_callable=AsyncMock, return_value="look at this"):
            with patch("jarvis.main.needs_vision", return_value=True):
                with patch("jarvis.main.capture", new_callable=AsyncMock, side_effect=RuntimeError("Cannot open camera")):
                    with patch("jarvis.main.speak", new_callable=AsyncMock) as mock_speak:
                        with patch("jarvis.main.think_and_act", new_callable=AsyncMock, return_value="Sure, here's what I think."):
                            await pipeline_iteration(interrupt, tools, conversation, settings)

    # Should have spoken the camera error AND then the response
    calls = mock_speak.call_args_list
    assert any("can't see" in str(c).lower() for c in calls)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_main.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write the implementation**

Create `src/jarvis/main.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_main.py -v
```

Expected: 4 passed

- [ ] **Step 5: Run all tests together**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass (approximately 20+ tests across all modules).

- [ ] **Step 6: Commit**

```bash
git add src/jarvis/main.py tests/test_main.py
git commit -m "feat: add main pipeline orchestration with interrupt support"
```

---

### Task 11: Integration Smoke Test

**Files:**
- Create: `.env` (local only, not committed)

- [ ] **Step 1: Create .env with your real API key**

```bash
cp .env.example .env
```

Then edit `.env` and set your real `DEEPSEEK_API_KEY`.

- [ ] **Step 2: Run Jarvis**

```bash
uv run jarvis
```

Expected output:
```
[Jarvis] Loading models and discovering shortcuts...
[Jarvis] Found N shortcuts.
[Jarvis] Listening for wake word... Say 'Hey Jarvis'!
```

- [ ] **Step 3: Test the full loop**

Say "Hey Jarvis" and then "what's on my calendar today". Verify:
- Wake word detection triggers recording
- Transcription produces text
- DeepSeek receives the text and calls the `What's on today?` shortcut
- Shortcut result feeds back to DeepSeek
- Response is spoken aloud

- [ ] **Step 4: Test vision**

Say "Hey Jarvis" and then "look at what's in front of me". Verify:
- Camera captures a frame
- Image is sent to DeepSeek along with text
- Response describes what the camera sees

- [ ] **Step 5: Test interrupt**

Say "Hey Jarvis", give a command, and while it's speaking the response, say "Hey Jarvis" again. Verify:
- TTS stops immediately
- New recording begins

- [ ] **Step 6: Final commit with any adjustments**

If any adjustments were needed during smoke testing, commit them:

```bash
git add -A
git commit -m "fix: adjustments from integration smoke testing"
```
