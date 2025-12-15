from __future__ import annotations

from typing import Iterable, Protocol
from app.domain.value_objects.types import LanguageCode


class IChatStreamer(Protocol):
    def stream(self, message: str, language: LanguageCode) -> Iterable[str]:
        ...

