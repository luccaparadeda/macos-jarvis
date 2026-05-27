import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from jarvis.ears import transcribe
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"anthropic_api_key": "k"}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_transcribe_returns_text():
    settings = _make_settings(whisper_model="mlx-community/whisper-tiny")
    audio = np.random.randn(16000).astype(np.float32)

    with patch("jarvis.ears.mlx_whisper") as mock_whisper:
        mock_whisper.transcribe.return_value = {"text": " Hello Jarvis "}
        result = await transcribe(audio, settings)

    assert result == "Hello Jarvis"
    mock_whisper.transcribe.assert_called_once()
    call_args = mock_whisper.transcribe.call_args
    assert call_args[1]["path_or_hf_repo"] == "mlx-community/whisper-tiny"


@pytest.mark.asyncio
async def test_transcribe_empty_audio():
    settings = _make_settings()
    audio = np.array([], dtype=np.float32)

    with patch("jarvis.ears.mlx_whisper") as mock_whisper:
        mock_whisper.transcribe.return_value = {"text": ""}
        result = await transcribe(audio, settings)

    assert result == ""
