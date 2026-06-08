import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.config import Settings
from jarvis.mouth import speak


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
            with patch("sounddevice.wait"):
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


def _fake_tts_result(samples, sample_rate=24000):
    result = MagicMock()
    result.audio = samples
    result.sample_rate = sample_rate
    return result


def test_generate_audio_concatenates_chunks():
    from jarvis.mouth import _generate_audio

    model = MagicMock()
    model.generate.return_value = iter(
        [
            _fake_tts_result([0.1] * 100),
            _fake_tts_result([0.2] * 50),
        ]
    )
    with patch("jarvis.mouth._get_model", return_value=model):
        audio, rate = _generate_audio("hello", _make_settings())

    assert rate == 24000
    assert len(audio) == 150


def test_generate_audio_empty_results():
    from jarvis.mouth import _generate_audio

    model = MagicMock()
    model.generate.return_value = iter([])
    with patch("jarvis.mouth._get_model", return_value=model):
        audio, rate = _generate_audio("hello", _make_settings())

    assert audio.size == 0
    assert rate == 24000


@pytest.mark.asyncio
async def test_speak_interrupt_mid_playback_stops_audio():
    settings = _make_settings()
    interrupt = asyncio.Event()
    one_second = np.zeros(24000, dtype=np.float32)

    async def set_interrupt_soon():
        await asyncio.sleep(0.15)
        interrupt.set()

    with patch("jarvis.mouth._generate_audio", return_value=(one_second, 24000)):
        with patch("sounddevice.play"):
            with patch("sounddevice.stop") as mock_stop:
                with patch("sounddevice.wait"):
                    await asyncio.gather(speak("a long sentence", interrupt, settings), set_interrupt_soon())

    mock_stop.assert_called_once()


@pytest.mark.asyncio
async def test_speak_empty_text_does_nothing():
    settings = _make_settings()
    interrupt = asyncio.Event()

    with patch("jarvis.mouth._generate_audio") as mock_gen:
        await speak("", interrupt, settings)

    mock_gen.assert_not_called()
