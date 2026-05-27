# Jarvis

A voice-controlled macOS assistant that runs on Apple Silicon. Say "Hey Jarvis" and it listens, thinks, acts, and talks back -- all with low latency and minimal memory usage.

Jarvis uses local models for audio (wake word, speech-to-text, text-to-speech) and Claude for reasoning and tool use. It controls your Mac through Apple Shortcuts, Spotlight search, and native file operations.

## How It Works

```
Wake Word → Record → Transcribe → [See] → Think → Act → Speak
```

| Component | What It Does | Model |
|-----------|-------------|-------|
| **Wake** | Always-on keyword detection | openWakeWord (`hey_jarvis`) |
| **Ears** | Speech-to-text | Whisper Small (MLX) |
| **Eyes** | Camera capture when you say "look at this" | OpenCV + Continuity Camera |
| **Brain** | Reasoning, conversation, tool routing | Claude (Anthropic API) |
| **Hands** | Runs Apple Shortcuts, opens files, searches Spotlight | macOS native |
| **Mouth** | Text-to-speech | Kokoro 82M (MLX) |

Audio processing stays entirely on-device. Only text prompts and camera frames are sent to the API.

## Requirements

- macOS 13.5+ on Apple Silicon (M1+)
- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)
- Microphone access

## Install

### One-liner

```bash
curl -fsSL https://raw.githubusercontent.com/luccaparadeda/macos-jarvis/main/install.sh | bash
```

### Homebrew

```bash
brew tap luccaparadeda/tap
brew install macos-jarvis
```

### PyPI

```bash
pip install macos-jarvis
```

### From source

```bash
git clone https://github.com/luccaparadeda/macos-jarvis.git
cd macos-jarvis
uv sync  # or: pip install -e .
```

Then set your API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create a `.env` file in the project directory:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
jarvis
```

That's it. Jarvis will calibrate your microphone, load models, discover your Apple Shortcuts, and start listening.

### Voice Commands

Just talk naturally after the wake word:

- **"Hey Jarvis, what time is it?"** -- answers via Claude
- **"Hey Jarvis, open my Downloads folder"** -- opens Finder
- **"Hey Jarvis, look at this and tell me what you see"** -- captures camera, sends image to Claude
- **"Hey Jarvis, run my Meeting Notes shortcut"** -- triggers an Apple Shortcut
- **"Hey Jarvis, find that invoice PDF"** -- searches via Spotlight

### Configuration

All settings are configurable via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Claude model to use |
| `WHISPER_MODEL` | `mlx-community/whisper-small-mlx` | Whisper model for STT |
| `KOKORO_MODEL` | `mlx-community/Kokoro-82M-bf16` | Kokoro model for TTS |
| `WAKE_MODEL` | `hey_jarvis` | Wake word model name |
| `CAMERA_INDEX` | `0` | Camera device index |
| `SILENCE_DURATION` | `1.5` | Seconds of silence before stopping recording |

## Architecture

```
src/jarvis/
├── main.py      # Pipeline orchestration
├── wake.py      # Wake word detection (openWakeWord)
├── audio.py     # Microphone recording with silence detection
├── ears.py      # Speech-to-text (MLX Whisper)
├── eyes.py      # Camera capture (OpenCV)
├── brain.py     # Claude API + tool execution loop
├── hands.py     # Apple Shortcuts, file ops, Spotlight search
├── mouth.py     # Text-to-speech (MLX Kokoro)
└── config.py    # Settings via pydantic-settings
```

The pipeline is fully async. Wake word detection runs on a background thread, and heavy model inference is offloaded to thread executors to keep the event loop responsive.

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for setup and guidelines.

## License

[MIT](LICENSE)
