import asyncio
import time

import numpy as np
import sounddevice as sd

from jarvis.config import Settings

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600
CALIBRATION_CHUNKS = 5


def _clamp_threshold(value: float, settings: Settings) -> float:
    return min(max(value, settings.silence_floor), settings.silence_ceiling)


async def record_until_silence(
    interrupt: asyncio.Event, settings: Settings, ambient: float | None = None
) -> np.ndarray:
    loop = asyncio.get_event_loop()
    chunks: list[np.ndarray] = []
    recording_done = asyncio.Event()
    start_time = time.monotonic()
    heard_voice = False
    last_voice_time = time.monotonic()

    calibration_samples: list[float] = []
    if settings.silence_threshold > 0:
        threshold = settings.silence_threshold
    elif ambient is not None:
        threshold = _clamp_threshold(ambient * settings.silence_multiplier, settings)
        print(f"[Audio] Threshold from idle ambient={ambient:.4f}: {threshold:.4f}")
    else:
        threshold = None  # calibrate from the first chunks of this stream

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
                ambient_est = np.mean(calibration_samples)
                threshold = _clamp_threshold(ambient_est * settings.silence_multiplier, settings)
                print(f"[Audio] Calibrated: ambient={ambient_est:.4f}, threshold={threshold:.4f}")
            return

        chunks.append(chunk)

        if amplitude > threshold:
            last_voice_time = now
            if not heard_voice:
                print(f"[Audio] Voice detected! (amplitude={amplitude:.4f})")
            heard_voice = True
        elif heard_voice and now - last_voice_time > settings.silence_duration:
            loop.call_soon_threadsafe(recording_done.set)
        elif not heard_voice and now - start_time > settings.no_voice_timeout:
            print(f"[Audio] No voice detected after {settings.no_voice_timeout:.0f}s, giving up.")
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

    if not heard_voice or not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)
