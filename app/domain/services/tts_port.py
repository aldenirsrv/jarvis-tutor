from __future__ import annotations

from typing import Generator, Iterable, Protocol
from app.domain.entities.audio import AudioFrame
from app.domain.value_objects.types import LanguageCode


class ITTSStreamer(Protocol):
    def stream_pcm(self, text_or_iter: str | Iterable[str], language: LanguageCode, quality: str) -> Generator[AudioFrame, None, None]:
        ...
