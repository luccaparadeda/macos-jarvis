import asyncio
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from openwakeword import Model

CHUNK_SIZE = 1280
SAMPLE_RATE = 16000
AMBIENT_WINDOW_CHUNKS = 50  # ~4s of recent chunk amplitudes
AMBIENT_MIN_CHUNKS = 10
AMBIENT_PERCENTILE = 20


class WakeWordListener:
    def __init__(
        self,
        wake_event: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
    ):
        self._wake_event = wake_event
        self._loop = loop
        self._model_name = model_name
        self._threshold = threshold
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._model: Model | None = None
        self._stream: sd.InputStream | None = None
        self._recent_amps: deque[float] = deque(maxlen=AMBIENT_WINDOW_CHUNKS)

    @property
    def ambient_level(self) -> float | None:
        """Noise floor estimated from idle listening (percentile is robust to bursts)."""
        if len(self._recent_amps) < AMBIENT_MIN_CHUNKS:
            return None
        return float(np.percentile(list(self._recent_amps), AMBIENT_PERCENTILE))

    def _track_ambient(self, amplitude: float) -> None:
        self._recent_amps.append(amplitude)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def pause(self) -> None:
        self._paused = True
        # Wait briefly for the listen loop to release the mic so the recorder
        # doesn't open a second stream on the same device (PaMacCore err -50).
        for _ in range(10):
            if self._stream is None or not self._stream.active:
                break
            time.sleep(0.05)

    def resume(self) -> None:
        self._paused = False
        if self._model:
            self._model.reset()

    def _listen_loop(self) -> None:
        self._model = Model(wakeword_models=[self._model_name], inference_framework="onnx")

        def audio_callback(indata, frames, time_info, status):
            if self._paused or not self._running:
                return

            chunk = indata[:, 0]
            self._track_ambient(float(np.abs(chunk).mean()))
            audio_data = (chunk * 32767).astype(np.int16)
            predictions = self._model.predict(audio_data)
            for key, score in predictions.items():
                if score > self._threshold:
                    print(f"[Wake] Detected! ({score:.2f})")
                    self._loop.call_soon_threadsafe(self._wake_event.set)

        def open_stream() -> sd.InputStream:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                dtype="float32",
                callback=audio_callback,
            )
            stream.start()
            return stream

        try:
            self._stream = open_stream()
            while self._running:
                # Release the mic while recording/speaking; reacquire on resume.
                # NOTE: always close + recreate — restarting a stopped stream
                # after another stream cycled the device yields a zombie stream
                # on macOS (PaMacCore err -50, zero callbacks delivered).
                if self._paused and self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
                    self._stream = None
                elif not self._paused and self._stream is None:
                    self._stream = open_stream()
                sd.sleep(100)
        except (KeyboardInterrupt, OSError):
            pass
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass


async def start_listener(
    wake_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
    model_name: str = "hey_jarvis",
    threshold: float = 0.5,
) -> WakeWordListener:
    listener = WakeWordListener(wake_event, loop, model_name, threshold)
    listener.start()
    return listener


async def stop_listener(listener: WakeWordListener) -> None:
    listener.stop()
