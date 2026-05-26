import asyncio
import time

import numpy as np
import sounddevice as sd

from jarvis.config import Settings

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600
MIN_RECORD_SECONDS = 0.5


async def record_until_silence(interrupt: asyncio.Event, settings: Settings) -> np.ndarray:
    loop = asyncio.get_event_loop()
    chunks: list[np.ndarray] = []
    recording_done = asyncio.Event()
    start_time = time.monotonic()
    heard_voice = False
    last_voice_time = time.monotonic()

    def callback(indata, frames, time_info, status):
        nonlocal last_voice_time, heard_voice
        if interrupt.is_set():
            loop.call_soon_threadsafe(recording_done.set)
            return

        chunk = indata[:, 0].copy()
        chunks.append(chunk)

        amplitude = np.abs(chunk).mean()
        now = time.monotonic()

        if amplitude > settings.silence_threshold:
            last_voice_time = now
            heard_voice = True
        elif heard_voice and now - last_voice_time > settings.silence_duration:
            loop.call_soon_threadsafe(recording_done.set)
        elif not heard_voice and now - start_time > 5.0:
            loop.call_soon_threadsafe(recording_done.set)

    if interrupt.is_set():
        return np.array([], dtype=np.float32)

    print("[Jarvis] Recording... speak now.")

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
