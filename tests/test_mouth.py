import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from jarvis.mouth import speak
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"anthropic_api_key": "k"}
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


def test_plain_speech_strips_markdown():
    from jarvis.mouth import _plain_speech
    assert _plain_speech("You have one item: **Buy milk**.") == "You have one item: Buy milk."
    assert _plain_speech("# Header\n- bullet one\n- bullet two") == "Header bullet one bullet two"
    assert _plain_speech("see [the docs](https://example.com) now") == "see the docs now"
    assert _plain_speech("normal to-do text stays") == "normal to-do text stays"


@pytest.mark.asyncio
async def test_speak_strips_markdown_before_tts():
    settings = _make_settings()
    interrupt = asyncio.Event()
    fake_audio = np.zeros(100, dtype=np.float32)

    with patch("jarvis.mouth._generate_audio", return_value=(fake_audio, 24000)) as mock_gen:
        with patch("sounddevice.play"):
            with patch("sounddevice.wait"):
                await speak("You have one item: **Buy milk**.", interrupt, settings)

    spoken = mock_gen.call_args[0][0]
    assert spoken == "You have one item: Buy milk."


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
