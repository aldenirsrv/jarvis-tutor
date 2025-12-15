import sqlite3

class SQLiteMemory:
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def add_message(self, role, content):
        # Garante que content é string simples
        if not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                content = "[Invalid content]"

        content = content.strip()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
        conn.commit()
        conn.close()

    def get_last_messages(self, limit=5):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT role, content FROM messages ORDER BY id DESC LIMIT ?', (limit,))
        rows = c.fetchall()
        conn.close()
        return list(reversed(rows))  # ordem cronológica