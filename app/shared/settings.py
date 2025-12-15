import os
from dataclasses import dataclass


@dataclass
class Settings:
    chat_backend: str
    hf_token: str | None
    tts_backend: str

    @classmethod
    def load(cls) -> "Settings":
        backend = os.getenv("CHAT_BACKEND", "huggingface").lower()
        tts_backend = os.getenv("TTS_BACKEND", "piper").lower()
        return cls(
            chat_backend=backend,
            hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            tts_backend=tts_backend,
        )
