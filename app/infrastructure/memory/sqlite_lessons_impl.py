import sqlite3
import uuid


class SQLiteLessons:
    def __init__(self, db_path: str = "lessons.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS lessons (
                id TEXT PRIMARY KEY,
                exercise_number TEXT NOT NULL,
                goals TEXT,
                instruction TEXT NOT NULL,
                content TEXT NOT NULL,
                rules TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('TO_STUDY','STUDYING','CONCLUDED')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

    def create_lesson(
        self,
        exercise_number: str,
        goals: str | None,
        instruction: str,
        content: str,
        rules: str,
        status: str,
    ) -> dict:
        lesson_id = str(uuid.uuid4())
        status_value = status.value if hasattr(status, "value") else str(status)
        status_value = status_value.upper()
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO lessons (id, exercise_number, goals, instruction, content, rules, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lesson_id,
                exercise_number,
                goals,
                instruction,
                content,
                rules,
                status_value,
            ),
        )
        conn.commit()
        conn.close()
        return {
            "id": lesson_id,
            "exercise_number": exercise_number,
            "goals": goals,
            "instruction": instruction,
            "content": content,
            "rules": rules,
            "status": status_value,
        }

    def list_lessons(self, status: str | None = None) -> list[dict]:
        status_value = None
        if status is not None:
            status_value = status.value if hasattr(status, "value") else str(status)
            status_value = status_value.upper()
        conn = self._connect()
        c = conn.cursor()
        if status_value:
            c.execute(
                """
                SELECT id, exercise_number, goals, instruction, content, rules, status
                FROM lessons
                WHERE status = ?
                ORDER BY created_at DESC
                """,
                (status_value,),
            )
        else:
            c.execute(
                """
                SELECT id, exercise_number, goals, instruction, content, rules, status
                FROM lessons
                ORDER BY created_at DESC
                """
            )
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_lesson(self, lesson_id: str) -> dict | None:
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """
            SELECT id, exercise_number, goals, instruction, content, rules, status
            FROM lessons
            WHERE id = ?
            """,
            (lesson_id,),
        )
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def delete_lesson(self, lesson_id: str) -> bool:
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))
        conn.commit()
        deleted = c.rowcount > 0
        conn.close()
        return deleted

    def update_status(self, lesson_id: str, status: str) -> dict | None:
        status_value = status.value if hasattr(status, "value") else str(status)
        status_value = status_value.upper()
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "UPDATE lessons SET status = ? WHERE id = ?",
            (status_value, lesson_id),
        )
        conn.commit()
        updated = c.rowcount > 0
        conn.close()
        if not updated:
            return None
        return self.get_lesson(lesson_id)
