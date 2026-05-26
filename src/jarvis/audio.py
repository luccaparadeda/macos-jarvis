import asyncio
import time

import numpy as np
import sounddevice as sd

from jarvis.config import Settings

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600


async def record_until_silence(interrupt: asyncio.Event, settings: Settings) -> np.ndarray:
    loop = asyncio.get_event_loop()
    chunks: list[np.ndarray] = []
    last_voice_time = time.monotonic()
    recording_done = asyncio.Event()

    def callback(indata, frames, time_info, status):
        nonlocal last_voice_time
        if interrupt.is_set():
            loop.call_soon_threadsafe(recording_done.set)
            return

        chunk = indata[:, 0].copy()
        chunks.append(chunk)

        amplitude = np.abs(chunk).mean()
        if amplitude > settings.silence_threshold:
            last_voice_time = time.monotonic()
        elif time.monotonic() - last_voice_time > settings.silence_duration:
            loop.call_soon_threadsafe(recording_done.set)

    if interrupt.is_set():
        return np.array([], dtype=np.float32)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=callback,
    ):
        await recording_done.wait()

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)
