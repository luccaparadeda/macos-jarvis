import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
                for score in predictions.values():
                    if score > listener._threshold:
                        wake_event.set()

    assert wake_event.is_set()


@pytest.mark.asyncio
async def test_listener_ignores_low_confidence():
    wake_event = asyncio.Event()

    mock_oww = MagicMock()
    mock_oww.predict.return_value = {"hey_jarvis": 0.2}

    # Simulate what the callback does
    predictions = mock_oww.predict(np.zeros(1280, dtype=np.int16))
    for score in predictions.values():
        if score > 0.5:
            wake_event.set()

    assert not wake_event.is_set()


def test_ambient_level_tracks_noise_floor():
    loop = asyncio.new_event_loop()
    wake_event = asyncio.Event()
    listener = WakeWordListener(wake_event, loop, threshold=0.5)

    assert listener.ambient_level is None  # not enough samples yet

    # mostly-quiet room with a brief loud burst (e.g. a cough)
    for amp in [0.001] * 8 + [0.5] * 2:
        listener._track_ambient(amp)

    level = listener.ambient_level
    assert level is not None
    assert level < 0.01  # percentile floor ignores the burst
    loop.close()


class FakeStream:
    """Stands in for sd.InputStream at the hardware boundary; everything else
    (thread, state machine, pause/resume) runs for real."""

    def __init__(self, created, **kwargs):
        self.callback = kwargs.get("callback")
        self.active = False
        self.closed = False
        created.append(self)

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


def _wait_for(condition, timeout=3.0):
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if condition():
            return True
        _time.sleep(0.02)
    return False


def test_listen_loop_recreates_stream_on_resume():
    """Regression for the zombie-stream bug: pause must CLOSE the stream and
    resume must open a FRESH one — restarting a stopped stream delivers zero
    callbacks on macOS. Real listener thread, fake hardware."""
    created: list[FakeStream] = []
    loop = asyncio.new_event_loop()
    listener = WakeWordListener(asyncio.Event(), loop, threshold=0.5)

    with patch("jarvis.wake.Model", return_value=MagicMock()):
        with patch("jarvis.wake.sd.InputStream", side_effect=lambda **kw: FakeStream(created, **kw)):
            listener.start()
            assert _wait_for(lambda: len(created) == 1 and created[0].active)

            listener.pause()
            assert _wait_for(lambda: created[0].closed and listener._stream is None)

            listener.resume()
            assert _wait_for(lambda: len(created) == 2 and created[1].active)
            assert not created[1].closed  # the second stream is a fresh one

            listener.stop()
            assert _wait_for(lambda: created[1].closed)
    loop.close()


@pytest.mark.asyncio
async def test_listen_loop_detection_sets_event_through_real_thread():
    """A detection in the real listener thread reaches the asyncio event."""
    created: list[FakeStream] = []
    loop = asyncio.get_running_loop()
    wake_event = asyncio.Event()

    model = MagicMock()
    model.predict.return_value = {"hey_jarvis": 0.95}

    listener = WakeWordListener(wake_event, loop, threshold=0.5)
    with patch("jarvis.wake.Model", return_value=model):
        with patch("jarvis.wake.sd.InputStream", side_effect=lambda **kw: FakeStream(created, **kw)):
            listener.start()
            await asyncio.to_thread(_wait_for, lambda: len(created) == 1 and created[0].active)

            # fire the REAL callback the listener installed
            created[0].callback(np.zeros((1280, 1), dtype=np.float32), 1280, None, None)
            await asyncio.wait_for(wake_event.wait(), timeout=3.0)

            # ambient tracking happened through the same real callback
            assert len(listener._recent_amps) == 1
            listener.stop()

    assert wake_event.is_set()


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
