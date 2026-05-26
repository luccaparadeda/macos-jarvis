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
