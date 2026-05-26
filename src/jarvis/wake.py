import asyncio
import threading

import numpy as np
import sounddevice as sd
from openwakeword import Model

CHUNK_SIZE = 1280
SAMPLE_RATE = 16000


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

    def resume(self) -> None:
        self._paused = False
        if self._model:
            self._model.reset()

    def _listen_loop(self) -> None:
        self._model = Model(wakeword_models=[self._model_name], inference_framework="onnx")

        def audio_callback(indata, frames, time_info, status):
            if self._paused or not self._running:
                return

            audio_data = (indata[:, 0] * 32767).astype(np.int16)
            predictions = self._model.predict(audio_data)
            for key, score in predictions.items():
                if score > self._threshold:
                    print(f"[Wake] Detected! ({score:.2f})")
                    self._loop.call_soon_threadsafe(self._wake_event.set)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                dtype="float32",
                callback=audio_callback,
            ):
                while self._running:
                    sd.sleep(100)
        except (KeyboardInterrupt, OSError):
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
