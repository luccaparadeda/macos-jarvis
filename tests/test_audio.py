import asyncio
import threading
import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from jarvis.audio import record_until_silence
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"anthropic_api_key": "k", "silence_threshold": 0.01, "silence_duration": 0.1}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_record_stops_on_silence():
    # silence_duration=0.05s, we'll send loud then silent with a 0.06s gap
    settings = _make_settings(silence_threshold=0.5, silence_duration=0.05)
    interrupt = asyncio.Event()

    loud_chunk = np.ones(1600, dtype=np.float32)
    silent_chunk = np.zeros(1600, dtype=np.float32)
    # loud first, then silent chunks with a delay so time.monotonic diff exceeds silence_duration
    chunks = [loud_chunk, silent_chunk, silent_chunk]

    def fake_input_stream(**kwargs):
        stream = MagicMock()
        callback = kwargs["callback"]

        def fire_callbacks():
            # fire loud chunk immediately
            callback(chunks[0].reshape(-1, 1), None, None, None)
            # wait longer than silence_duration before sending silent chunks
            time.sleep(0.08)
            callback(chunks[1].reshape(-1, 1), None, None, None)
            callback(chunks[2].reshape(-1, 1), None, None, None)

        def enter(self):
            t = threading.Thread(target=fire_callbacks, daemon=True)
            t.start()
            return stream

        stream.__enter__ = enter
        stream.__exit__ = MagicMock(return_value=False)
        return stream

    with patch("sounddevice.InputStream", side_effect=fake_input_stream):
        result = await record_until_silence(interrupt, settings)

    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def _stream_firing(chunk_sequence, gap_after_first=0.08):
    """Fake InputStream that fires the given chunks, sleeping after the first."""
    def fake_input_stream(**kwargs):
        stream = MagicMock()
        callback = kwargs["callback"]

        def fire_callbacks():
            callback(chunk_sequence[0].reshape(-1, 1), None, None, None)
            time.sleep(gap_after_first)
            for chunk in chunk_sequence[1:]:
                callback(chunk.reshape(-1, 1), None, None, None)

        def enter(self):
            t = threading.Thread(target=fire_callbacks, daemon=True)
            t.start()
            return stream

        stream.__enter__ = enter
        stream.__exit__ = MagicMock(return_value=False)
        return stream
    return fake_input_stream


@pytest.mark.asyncio
async def test_ambient_param_skips_calibration():
    # Idle ambient provided: no calibration window, first chunk is captured
    settings = _make_settings(silence_threshold=0.0, silence_duration=0.05)
    interrupt = asyncio.Event()
    voiced = np.full(1600, 0.05, dtype=np.float32)
    quiet = np.zeros(1600, dtype=np.float32)

    with patch("sounddevice.InputStream", side_effect=_stream_firing([voiced, quiet])):
        result = await record_until_silence(interrupt, settings, ambient=0.0004)

    # ambient*3 clamps up to floor (0.01); voiced chunk (0.05) detected from chunk one
    assert len(result) == 3200  # both chunks kept — nothing eaten by calibration


@pytest.mark.asyncio
async def test_threshold_ceiling_keeps_speech_detectable():
    # Inflated ambient (background media) must not produce an unreachable threshold
    settings = _make_settings(silence_threshold=0.0, silence_duration=0.05)
    interrupt = asyncio.Event()
    speech = np.full(1600, 0.09, dtype=np.float32)
    quiet = np.zeros(1600, dtype=np.float32)

    with patch("sounddevice.InputStream", side_effect=_stream_firing([speech, quiet])):
        result = await record_until_silence(interrupt, settings, ambient=0.0481)

    # 0.0481*3 = 0.1443 uncapped (unreachable); ceiling caps at 0.08 < 0.09 speech
    assert len(result) > 0


@pytest.mark.asyncio
async def test_no_voice_returns_empty():
    # Give-up path must not ship garbage audio to STT
    settings = _make_settings(silence_threshold=0.5, no_voice_timeout=0.05)
    interrupt = asyncio.Event()
    quiet1 = np.zeros(1600, dtype=np.float32)
    quiet2 = np.zeros(1600, dtype=np.float32)

    with patch("sounddevice.InputStream", side_effect=_stream_firing([quiet1, quiet2], gap_after_first=0.07)):
        result = await asyncio.wait_for(record_until_silence(interrupt, settings), timeout=2.0)

    assert result.size == 0


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
