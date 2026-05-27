# Contributing

Thanks for your interest in contributing to Jarvis! This project is in its early stages and contributions of all kinds are welcome.

## Getting Started

1. Fork the repo and clone it locally
2. Set up the development environment:

```bash
uv sync --group dev
```

3. Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

4. Run the tests:

```bash
uv run pytest
```

## Making Changes

1. Create a branch from `main`
2. Make your changes
3. Run the tests to make sure nothing is broken
4. Commit with a descriptive message (e.g. `fix: resolve camera timeout on M1`, `feat: add volume control`)
5. Open a pull request against `main`

## Project Structure

```
src/jarvis/
├── main.py      # Pipeline orchestration
├── wake.py      # Wake word detection
├── audio.py     # Microphone recording
├── ears.py      # Speech-to-text (MLX Whisper)
├── eyes.py      # Camera capture (OpenCV)
├── brain.py     # Claude API + tool execution
├── hands.py     # Apple Shortcuts, file ops, Spotlight
├── mouth.py     # Text-to-speech (MLX Kokoro)
└── config.py    # Settings (pydantic-settings)
```

Each module maps to a "sense" or capability. If you're adding a new tool the brain can call, you'll work in `hands.py` (schema + execution) and `brain.py` (tool dispatch).

## What To Work On

- Check [open issues](https://github.com/luccaparadeda/macos-jarvis/issues) for bugs and feature requests
- Performance improvements for 8GB machines are always welcome
- Better error handling and edge cases in audio recording
- New tool integrations (AppleScript, Automator, system commands)

## Guidelines

- Keep it simple. This project values small, focused modules over abstractions.
- All audio processing must stay local. Only text and camera frames go to the API.
- Test on Apple Silicon -- this project doesn't target Intel Macs.
- If you're unsure about a change, open an issue first to discuss it.

## Code Style

- No strict formatter enforced yet -- just keep it consistent with the existing code
- Type hints are encouraged
- Async by default for anything that does I/O

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](../LICENSE).
