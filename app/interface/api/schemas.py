from enum import Enum
from pydantic import BaseModel
from app.domain.value_objects.types import AudioFormat, LanguageCode, LanguageISO
from typing import Optional

class TTSRequest(BaseModel):
    message: str
    language: str
    quality: str
    selected_lesson: Optional[str]
    format: str | None = None  # default handled in route

    def language_code(self) -> LanguageCode:
        return LanguageCode.from_str(self.language)
    
    def language_iso(self) -> LanguageCode:
        return LanguageISO.from_str(self.language)


    def audio_format(self) -> AudioFormat:
        # Default to WAV if not provided
        return AudioFormat.from_str(self.format or "wav")


class LessonStatus(str, Enum):
    TO_STUDY = "TO_STUDY"
    STUDYING = "STUDYING"
    CONCLUDED = "CONCLUDED"


class LessonCreate(BaseModel):
    exercise_number: str
    goals: str | None = None
    instruction: str
    content: str
    rules: str
    status: LessonStatus = LessonStatus.TO_STUDY


class Lesson(BaseModel):
    id: str
    exercise_number: str
    goals: str | None = None
    instruction: str
    content: str
    rules: str
    status: LessonStatus


class LessonStatusUpdate(BaseModel):
    status: LessonStatus
