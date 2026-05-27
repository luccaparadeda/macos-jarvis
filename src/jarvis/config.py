from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-haiku-4-5-20251001"
    whisper_model: str = "mlx-community/whisper-small-mlx"
    kokoro_model: str = "mlx-community/Kokoro-82M-bf16"
    wake_model: str = "hey_jarvis"
    camera_index: int = 0
    silence_threshold: float = 0.0
    silence_multiplier: float = 3.0
    silence_duration: float = 1.5
    vision_keywords: list[str] = [
        "look",
        "see",
        "show",
        "what is",
        "camera",
        "screen",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
