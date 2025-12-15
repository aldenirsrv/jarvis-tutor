from pydantic import BaseModel
from app.domain.value_objects.types import AudioFormat, LanguageCode, LanguageISO


class TTSRequest(BaseModel):
    message: str
    language: str
    format: str | None = None  # default handled in route

    def language_code(self) -> LanguageCode:
        return LanguageCode.from_str(self.language)
    
    def language_iso(self) -> LanguageCode:
        return LanguageISO.from_str(self.language)


    def audio_format(self) -> AudioFormat:
        # Default to WAV if not provided
        return AudioFormat.from_str(self.format or "wav")
