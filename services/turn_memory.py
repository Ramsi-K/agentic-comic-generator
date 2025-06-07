from datetime import datetime
import sqlite3


# Define a lightweight SQLite-backed memory manager for multi-agent state tracking
class AgentMemory:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    agent_name TEXT,
                    step_type TEXT,
                    content TEXT,
                    timestamp TEXT
                )
            """
            )

    def add_message(
        self, session_id, agent_name, content, step_type="message"
    ):
        timestamp = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO memory (session_id, agent_name, step_type, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (session_id, agent_name, step_type, content, timestamp),
            )

    def get_session_history(self, session_id):
        with self.conn:
            cur = self.conn.execute(
                """
                SELECT agent_name, step_type, content, timestamp
                FROM memory
                WHERE session_id = ?
                ORDER BY timestamp
            """,
                (session_id,),
            )
            return cur.fetchall()

    def get_as_llama_format(self, session_id):
        history = self.get_session_history(session_id)
        return [{"role": row[0], "content": row[2]} for row in history]


# Example usage
memory = AgentMemory(
    ":memory:"
)  # For in-memory DB; use 'memory.db' for persistent
memory.add_message(
    "sess-123", "AgentBrown", "Hello, I'm initiating the comic task.", "prompt"
)
memory.add_message(
    "sess-123", "AgentBayko", "Here's the first draft panel.", "response"
)

# Return a preview of stored memory
memory.get_session_history("sess-123")
