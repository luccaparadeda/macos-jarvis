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
