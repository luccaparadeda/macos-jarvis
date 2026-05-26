import asyncio
import threading

from openwakeword import Model
from pyaudio import PyAudio

CHUNK_SIZE = 1280
SAMPLE_RATE = 16000
FORMAT_WIDTH = 2


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
        self._running = True
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _listen_loop(self) -> None:
        model = Model(wakeword_models=[self._model_name])
        pa = PyAudio()
        stream = pa.open(
            rate=SAMPLE_RATE,
            channels=1,
            format=pa.get_format_from_width(FORMAT_WIDTH),
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        try:
            while self._running:
                audio_data = stream.read(CHUNK_SIZE)
                predictions = model.predict(audio_data)
                for key, score in predictions.items():
                    if score > self._threshold:
                        self._wake_event._value = True  # noqa: SLF001
                        self._loop.call_soon_threadsafe(self._wake_event.set)
        except (KeyboardInterrupt, OSError):
            pass
        finally:
            stream.close()
            pa.terminate()


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
