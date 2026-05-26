import asyncio
import functools

import mlx_whisper
import numpy as np

from jarvis.config import Settings


async def transcribe(audio: np.ndarray, settings: Settings) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        functools.partial(
            mlx_whisper.transcribe,
            audio,
            path_or_hf_repo=settings.whisper_model,
        ),
    )
    return result["text"].strip()
