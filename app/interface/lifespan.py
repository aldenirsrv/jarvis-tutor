from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.application.policies.chunking import buffered
from app.application.policies.pre_processor import add_natural_pauses
from app.application.use_cases.stream_tts import StreamTTSUseCase
from app.infrastructure.chat.hf_chat import HuggingFaceChatAdapter
from app.infrastructure.chat.openai_chat import OpenAIChatAdapter
from app.infrastructure.config.voice_registry import build_registry
from app.infrastructure.memory.sqlite_memory import SQLiteMemoryAdapter
from app.infrastructure.nltk.nltk_resources import ensure_nltk_punkt
from app.infrastructure.tts.multi_streamer import MultiTTSAdapter
from app.infrastructure.tts.piper_streamer import PiperTTSAdapter
from app.infrastructure.tts.silero_streamer import SileroTTSAdapter
from app.shared.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.load()
    memory = SQLiteMemoryAdapter()
    ensure_nltk_punkt()
    registry = build_registry()
    piper = PiperTTSAdapter(registry)
    silero = None
    if settings.tts_backend == "silero":
        try:
            silero = SileroTTSAdapter()
        except Exception as exc:
            print(f"[lifespan] Silero init failed ({exc}); falling back to Piper")
    tts = MultiTTSAdapter(piper, silero)

    if settings.chat_backend == "openai":
        chat = OpenAIChatAdapter(memory)
    else:
        chat = HuggingFaceChatAdapter(memory, hf_token=settings.hf_token)

    use_case = StreamTTSUseCase(chat=chat, tts=tts, chunker=buffered)

    app.state.settings = settings
    app.state.voice_registry = registry
    app.state.chat = chat
    app.state.stream_tts = use_case
    try:
        yield
    finally:
        pass
