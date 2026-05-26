import base64
import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from jarvis.eyes import capture
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"deepseek_api_key": "k", "camera_index": 0}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_capture_returns_base64_jpeg():
    import cv2
    settings = _make_settings()
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Create a real JPEG buffer before patching cv2.imencode
    _, real_buf = cv2.imencode(".jpg", fake_frame)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, fake_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc:
        with patch("cv2.imencode") as mock_enc:
            mock_enc.return_value = (True, real_buf)
            result = await capture(settings)

    mock_vc.assert_called_once_with(0)
    mock_cap.release.assert_called_once()

    decoded = base64.b64decode(result)
    assert len(decoded) > 0


@pytest.mark.asyncio
async def test_capture_raises_on_camera_failure():
    settings = _make_settings()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(RuntimeError, match="Cannot open camera"):
            await capture(settings)
