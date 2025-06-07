"""
Unified Memory System
Combines active conversation memory with persistent SQLite storage
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Any


class AgentMemory:
    """Unified memory system with both active conversation and persistent storage"""

    def __init__(
        self, session_id: str, agent_name: str, db_path: str = "memory.db"
    ):
        self.session_id = session_id
        self.agent_name = agent_name
        self.db_path = db_path

        # Active conversation memory (in-memory for fast access)
        self.active_messages = []

        # Initialize SQLite for persistence
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        """Create SQLite table for persistent storage"""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    agent_name TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT
                )
                """
            )

    def add_message(self, role: str, content: str):
        """Add message to both active memory and persistent storage"""
        timestamp = datetime.utcnow().isoformat()

        # Add to active memory
        message = {"role": role, "content": content, "timestamp": timestamp}
        self.active_messages.append(message)

        # Add to persistent storage
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO memory (session_id, agent_name, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.session_id, self.agent_name, role, content, timestamp),
            )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get active conversation history"""
        return self.active_messages.copy()

    def get_session_history(
        self, session_id: str = None
    ) -> List[Dict[str, Any]]:
        """Get persistent session history from SQLite"""
        if session_id is None:
            session_id = self.session_id

        with self.conn:
            cur = self.conn.execute(
                """
                SELECT role, content, timestamp
                FROM memory
                WHERE session_id = ?
                ORDER BY timestamp
                """,
                (session_id,),
            )
            rows = cur.fetchall()

        return [
            {"role": row[0], "content": row[1], "timestamp": row[2]}
            for row in rows
        ]

    def clear(self):
        """Clear active memory (keeps persistent storage)"""
        self.active_messages.clear()

    def clear_session(self, session_id: str = None):
        """Clear persistent storage for a session"""
        if session_id is None:
            session_id = self.session_id

        with self.conn:
            self.conn.execute(
                "DELETE FROM memory WHERE session_id = ?", (session_id,)
            )

    def get_memory_size(self) -> int:
        """Get size of active memory"""
        return len(self.active_messages)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
