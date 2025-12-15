from __future__ import annotations

from typing import Iterable, Tuple
from app.domain.services.memory_port import IMemory
from app.infrastructure.memory.sqlite_memory_impl import SQLiteMemory


class SQLiteMemoryAdapter(IMemory):
    def __init__(self, db_path: str = "conversations.db"):
        self._backend = SQLiteMemory(db_path)

    def add_message(self, role: str, content: str) -> None:
        self._backend.add_message(role, content)

    def get_last_messages(self, limit: int = 5) -> Iterable[Tuple[str, str]]:
        return self._backend.get_last_messages(limit=limit)
