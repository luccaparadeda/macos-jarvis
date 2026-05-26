import asyncio
import pytest
from unittest.mock import patch, MagicMock

from jarvis.wake import WakeWordListener


@pytest.mark.asyncio
async def test_listener_sets_event_on_detection():
    loop = asyncio.get_event_loop()
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

            import threading
            t = threading.Thread(target=listener._listen_loop, daemon=True)
            t.start()
            t.join(timeout=1.0)

    assert wake_event.is_set()


def test_listener_ignores_low_confidence():
    loop = asyncio.new_event_loop()
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

            import threading
            t = threading.Thread(target=listener._listen_loop, daemon=True)
            t.start()
            t.join(timeout=1.0)

    assert not wake_event.is_set()
    loop.close()
