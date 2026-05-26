from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    whisper_model: str = "mlx-community/whisper-tiny"
    kokoro_model: str = "mlx-community/Kokoro-82M-bf16"
    wake_model: str = "hey_jarvis"
    camera_index: int = 0
    silence_threshold: float = 0.01
    silence_duration: float = 1.5
    vision_keywords: list[str] = [
        "look",
        "see",
        "show",
        "what is",
        "camera",
        "screen",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
