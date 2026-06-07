import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jarvis.main import pipeline_iteration
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"anthropic_api_key": "test-key"}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_pipeline_text_only():
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
    assert call_args[0][1] == "base64img"


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

    calls = mock_speak.call_args_list
    assert any("can't see" in str(c).lower() for c in calls)


@pytest.mark.asyncio
async def test_pipeline_pauses_listener_during_tts():
    # Wake listener must not hear Jarvis's own voice (self-trigger + ambient pollution)
    settings = _make_settings()
    interrupt = asyncio.Event()
    audio_buf = np.random.randn(16000).astype(np.float32)
    listener = MagicMock()
    listener.ambient_level = 0.001

    with patch("jarvis.main.record_until_silence", new_callable=AsyncMock, return_value=audio_buf):
        with patch("jarvis.main.transcribe", new_callable=AsyncMock, return_value="hello"):
            with patch("jarvis.main.needs_vision", return_value=False):
                with patch("jarvis.main.think_and_act", new_callable=AsyncMock, return_value="Hi!"):
                    with patch("jarvis.main.speak", new_callable=AsyncMock):
                        await pipeline_iteration(interrupt, [], [], settings, listener)

    # paused for recording AND for TTS playback
    assert listener.pause.call_count == 2
    assert listener.resume.call_count == 2


@pytest.mark.asyncio
async def test_pipeline_passes_listener_ambient_to_recorder():
    settings = _make_settings()
    interrupt = asyncio.Event()
    audio_buf = np.random.randn(16000).astype(np.float32)
    listener = MagicMock()
    listener.ambient_level = 0.003

    with patch("jarvis.main.record_until_silence", new_callable=AsyncMock, return_value=audio_buf) as mock_record:
        with patch("jarvis.main.transcribe", new_callable=AsyncMock, return_value="hi"):
            with patch("jarvis.main.needs_vision", return_value=False):
                with patch("jarvis.main.think_and_act", new_callable=AsyncMock, return_value="Hello!"):
                    with patch("jarvis.main.speak", new_callable=AsyncMock):
                        await pipeline_iteration(interrupt, [], [], settings, listener)

    assert mock_record.call_args[1]["ambient"] == 0.003


@pytest.mark.asyncio
async def test_pipeline_empty_audio_short_circuits():
    settings = _make_settings()
    interrupt = asyncio.Event()
    empty_buf = np.array([], dtype=np.float32)

    with patch("jarvis.main.record_until_silence", new_callable=AsyncMock, return_value=empty_buf):
        with patch("jarvis.main.transcribe", new_callable=AsyncMock) as mock_transcribe:
            with patch("jarvis.main.speak", new_callable=AsyncMock) as mock_speak:
                await pipeline_iteration(interrupt, [], [], settings)

    mock_transcribe.assert_not_called()
    mock_speak.assert_called_once_with("I didn't catch that.", interrupt, settings)


@pytest.mark.asyncio
async def test_pipeline_passes_system_extra_to_brain():
    settings = _make_settings()
    interrupt = asyncio.Event()
    conversation: list[dict] = []
    tools = []
    audio_buf = np.random.randn(16000).astype(np.float32)

    with patch("jarvis.main.record_until_silence", new_callable=AsyncMock, return_value=audio_buf):
        with patch("jarvis.main.transcribe", new_callable=AsyncMock, return_value="add milk to my todos"):
            with patch("jarvis.main.needs_vision", return_value=False):
                with patch("jarvis.main.think_and_act", new_callable=AsyncMock, return_value="Done.") as mock_think:
                    with patch("jarvis.main.speak", new_callable=AsyncMock):
                        await pipeline_iteration(
                            interrupt, tools, conversation, settings,
                            system_extra="## Your memories\n(none)",
                        )

    assert mock_think.call_args[1]["system_extra"] == "## Your memories\n(none)"
