from __future__ import annotations

from dataclasses import dataclass
from app.domain.value_objects.types import AudioFormat, LanguageCode, LanguageISO


@dataclass
class TTSRequestDTO:
    message: str
    language: LanguageCode
    language_iso:LanguageISO
    format: AudioFormat = AudioFormat.WAV

