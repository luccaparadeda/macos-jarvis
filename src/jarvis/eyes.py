import asyncio
import base64
import functools

import cv2
import numpy as np

from jarvis.config import Settings


def _capture_sync(camera_index: int) -> str:
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        success, buf = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")

        return base64.b64encode(buf.tobytes()).decode("utf-8")
    finally:
        cap.release()


async def capture(settings: Settings) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(_capture_sync, settings.camera_index),
    )
