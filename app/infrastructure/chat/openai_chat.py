from __future__ import annotations

from typing import Iterable
from app.domain.services.chat_port import IChatStreamer
from app.domain.services.memory_port import IMemory
from app.domain.value_objects.types import LanguageCode
from app.infrastructure.chat.models import OpenAIChat


class OpenAIChatAdapter(IChatStreamer):
    def __init__(self, memory: IMemory):
        self._client = OpenAIChat(memory=memory)

    def stream(self, message: str, language: LanguageCode,lesson:str = None) -> Iterable[str]:
        return self._client.stream(message, language, lesson)
