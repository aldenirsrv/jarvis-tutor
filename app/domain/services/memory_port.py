from __future__ import annotations

from typing import Iterable, Protocol, Tuple


class IMemory(Protocol):
    def add_message(self, role: str, content: str) -> None:
        ...

    def get_last_messages(self, limit: int = 5) -> Iterable[Tuple[str, str]]:
        ...
