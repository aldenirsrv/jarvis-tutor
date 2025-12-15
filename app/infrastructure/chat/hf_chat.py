from __future__ import annotations

from typing import Iterable
from app.domain.services.chat_port import IChatStreamer
from app.domain.services.memory_port import IMemory
from app.domain.value_objects.types import LanguageCode
from app.infrastructure.chat.models import HuggingFaceChat


class HuggingFaceChatAdapter(IChatStreamer):
    def __init__(self, memory: IMemory, hf_token: str | None = None):
        self._client = HuggingFaceChat(memory=memory, hf_token=hf_token)

    def stream(self, message: str, language: LanguageCode) -> Iterable[str]:
        return self._client.stream(message, language)
