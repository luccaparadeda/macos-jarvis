import asyncio
import pytest
from unittest.mock import patch, MagicMock

from jarvis.wake import WakeWordListener


@pytest.mark.asyncio
async def test_listener_sets_event_on_detection():
    loop = asyncio.get_running_loop()
    wake_event = asyncio.Event()

    mock_oww = MagicMock()
    prediction = {"hey_jarvis": 0.9}
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
            listener.start()

            # Give the thread time to run and call_soon_threadsafe to be processed
            await asyncio.sleep(0.2)

    assert wake_event.is_set()


@pytest.mark.asyncio
async def test_listener_ignores_low_confidence():
    loop = asyncio.get_running_loop()
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
            listener.start()

            await asyncio.sleep(0.2)

    assert not wake_event.is_set()
