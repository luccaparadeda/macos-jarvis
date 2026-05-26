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

    chunk_count = 0

    def callback(indata, frames, time_info, status):
        nonlocal last_voice_time, heard_voice, chunk_count
        chunk_count += 1
        if interrupt.is_set():
            loop.call_soon_threadsafe(recording_done.set)
            return

        chunk = indata[:, 0].copy()
        chunks.append(chunk)

        amplitude = np.abs(chunk).mean()
        now = time.monotonic()

        if chunk_count <= 3 or (chunk_count % 50 == 0):
            print(f"[Audio] chunk {chunk_count}: amplitude={amplitude:.6f} threshold={settings.silence_threshold}")

        if amplitude > settings.silence_threshold:
            last_voice_time = now
            if not heard_voice:
                print(f"[Audio] Voice detected! amplitude={amplitude:.4f}")
            heard_voice = True
        elif heard_voice and now - last_voice_time > settings.silence_duration:
            print(f"[Audio] Silence detected, stopping recording.")
            loop.call_soon_threadsafe(recording_done.set)
        elif not heard_voice and now - start_time > 5.0:
            print(f"[Audio] No voice detected after 5s, giving up.")
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
