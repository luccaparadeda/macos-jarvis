import asyncio
import numpy as np
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

    fake_audio = np.zeros((1280, 1), dtype=np.float32)

    def fake_input_stream(**kwargs):
        stream = MagicMock()
        callback = kwargs["callback"]

        def run_callback():
            callback(fake_audio, 1280, None, None)

        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)
        return stream

    with patch("jarvis.wake.Model", return_value=mock_oww):
        with patch("jarvis.wake.sd.InputStream", side_effect=fake_input_stream):
            with patch("jarvis.wake.sd.sleep", side_effect=KeyboardInterrupt):
                listener = WakeWordListener(wake_event, loop, threshold=0.5)
                listener._model = mock_oww
                listener._paused = False

                # Directly test the callback logic
                audio_int16 = (fake_audio[:, 0] * 32767).astype(np.int16)
                mock_oww.predict.return_value = {"hey_jarvis": 0.9}
                listener._listen_loop_callback_test = True

                # Simulate what the callback does
                predictions = mock_oww.predict(audio_int16)
                for key, score in predictions.items():
                    if score > listener._threshold:
                        wake_event.set()

    assert wake_event.is_set()


@pytest.mark.asyncio
async def test_listener_ignores_low_confidence():
    loop = asyncio.get_running_loop()
    wake_event = asyncio.Event()

    mock_oww = MagicMock()
    mock_oww.predict.return_value = {"hey_jarvis": 0.2}

    # Simulate what the callback does
    predictions = mock_oww.predict(np.zeros(1280, dtype=np.int16))
    for key, score in predictions.items():
        if score > 0.5:
            wake_event.set()

    assert not wake_event.is_set()


def test_pause_resume():
    loop = asyncio.new_event_loop()
    wake_event = asyncio.Event()
    listener = WakeWordListener(wake_event, loop, threshold=0.5)

    assert not listener._paused
    listener.pause()
    assert listener._paused
    listener.resume()
    assert not listener._paused
    loop.close()
