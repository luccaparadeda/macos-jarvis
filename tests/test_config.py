import os
from jarvis.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
    settings = Settings()
    assert settings.deepseek_api_key == "test-key-123"
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-chat"


def test_settings_defaults():
    settings = Settings(deepseek_api_key="k")
    assert settings.whisper_model == "mlx-community/whisper-tiny"
    assert settings.kokoro_model == "mlx-community/Kokoro-82M-bf16"
    assert settings.wake_model == "hey_jarvis"
    assert settings.camera_index == 0
    assert settings.silence_threshold == 0.0
    assert settings.silence_multiplier == 3.0
    assert settings.silence_duration == 1.5
    assert "look" in settings.vision_keywords
    assert "see" in settings.vision_keywords


def test_settings_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("CAMERA_INDEX", "2")
    settings = Settings()
    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.camera_index == 2
