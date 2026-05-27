import os
from jarvis.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    settings = Settings()
    assert settings.anthropic_api_key == "test-key-123"
    assert settings.anthropic_model == "claude-haiku-4-5-20251001"


def test_settings_defaults():
    settings = Settings(anthropic_api_key="k")
    assert settings.whisper_model == "mlx-community/whisper-small-mlx"
    assert settings.kokoro_model == "mlx-community/Kokoro-82M-bf16"
    assert settings.wake_model == "hey_jarvis"
    assert settings.camera_index == 0
    assert settings.silence_threshold == 0.0
    assert settings.silence_multiplier == 3.0
    assert settings.silence_duration == 1.5
    assert "look" in settings.vision_keywords
    assert "see" in settings.vision_keywords


def test_settings_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("CAMERA_INDEX", "2")
    settings = Settings()
    assert settings.anthropic_model == "claude-haiku-4-5-20251001"
    assert settings.camera_index == 2
