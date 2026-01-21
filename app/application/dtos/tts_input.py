from __future__ import annotations

from dataclasses import dataclass
from app.domain.value_objects.types import AudioFormat, LanguageCode, LanguageISO
from typing import Optional

@dataclass
class TTSRequestDTO:
    message: str
    quality: str
    selected_lesson: Optional[str]
    language: LanguageCode
    language_iso:LanguageISO
    format: AudioFormat = AudioFormat.WAV

