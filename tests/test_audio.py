import asyncio
import threading
import time
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
