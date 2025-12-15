from __future__ import annotations

from dataclasses import dataclass
from app.domain.value_objects.types import LanguageCode


@dataclass
class Message:
    content: str
    language: LanguageCode


@dataclass
class VoiceProfile:
    language: LanguageCode
    voice_id: str | None = None

