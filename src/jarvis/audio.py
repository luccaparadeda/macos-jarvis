import asyncio
import time

import numpy as np
import sounddevice as sd

from jarvis.config import Settings

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600
CALIBRATION_CHUNKS = 5


async def record_until_silence(interrupt: asyncio.Event, settings: Settings) -> np.ndarray:
    loop = asyncio.get_event_loop()
    chunks: list[np.ndarray] = []
    recording_done = asyncio.Event()
    start_time = time.monotonic()
    heard_voice = False
    last_voice_time = time.monotonic()

    calibration_samples: list[float] = []
    threshold = settings.silence_threshold if settings.silence_threshold > 0 else None

    def callback(indata, frames, time_info, status):
        nonlocal last_voice_time, heard_voice, threshold
        if interrupt.is_set():
            loop.call_soon_threadsafe(recording_done.set)
            return

        chunk = indata[:, 0].copy()
        amplitude = np.abs(chunk).mean()
        now = time.monotonic()

        if threshold is None:
            calibration_samples.append(amplitude)
            if len(calibration_samples) >= CALIBRATION_CHUNKS:
                ambient = np.mean(calibration_samples)
                threshold = ambient * settings.silence_multiplier
                print(f"[Audio] Calibrated: ambient={ambient:.4f}, threshold={threshold:.4f}")
            return

        chunks.append(chunk)

        if amplitude > threshold:
            last_voice_time = now
            if not heard_voice:
                print(f"[Audio] Voice detected! (amplitude={amplitude:.4f})")
            heard_voice = True
        elif heard_voice and now - last_voice_time > settings.silence_duration:
            loop.call_soon_threadsafe(recording_done.set)
        elif not heard_voice and now - start_time > 5.0:
            print("[Audio] No voice detected after 5s, giving up.")
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
