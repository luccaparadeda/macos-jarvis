import asyncio
import functools

import numpy as np
import sounddevice as sd

from jarvis.config import Settings

_tts_pipeline = None


def _get_tts_pipeline(settings: Settings):
    global _tts_pipeline
    if _tts_pipeline is None:
        from mlx_audio.tts import TTS
        _tts_pipeline = TTS("mlx-community/Kokoro-82M-bf16")
    return _tts_pipeline


def _generate_audio(text: str, settings: Settings) -> tuple[np.ndarray, int]:
    tts = _get_tts_pipeline(settings)
    result = tts.generate(text=text)
    audio = result["audio"]
    sample_rate = result["sample_rate"]
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    return audio, sample_rate


async def speak(text: str, interrupt: asyncio.Event, settings: Settings) -> None:
    if not text or interrupt.is_set():
        return

    loop = asyncio.get_event_loop()
    audio, sample_rate = await loop.run_in_executor(
        None,
        functools.partial(_generate_audio, text, settings),
    )

    if interrupt.is_set():
        return

    sd.play(audio, samplerate=sample_rate)

    duration = len(audio) / sample_rate
    poll_interval = 0.1
    elapsed = 0.0
    while elapsed < duration:
        if interrupt.is_set():
            sd.stop()
            return
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    sd.wait()
