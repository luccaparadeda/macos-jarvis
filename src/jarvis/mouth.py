import asyncio
import functools

import numpy as np
import sounddevice as sd

from jarvis.config import Settings

_model = None


def _get_model(settings: Settings):
    global _model
    if _model is None:
        from mlx_audio.tts import load_model
        _model = load_model(settings.kokoro_model)
    return _model


def _generate_audio(text: str, settings: Settings) -> tuple[np.ndarray, int]:
    model = _get_model(settings)
    results = list(model.generate(text))
    if not results:
        return np.array([], dtype=np.float32), 24000

    audio_chunks = []
    sample_rate = 24000
    for r in results:
        sample_rate = r.sample_rate
        audio_chunks.append(np.array(r.audio, dtype=np.float32))

    audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
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
