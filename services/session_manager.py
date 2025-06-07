"""
Session Manager Service
Handles file I/O operations for session state management.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict

from services.message_factory import AgentMessage

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session state persistence and file operations"""

    def __init__(self, session_id: str, conversation_id: str):
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.session_dir = Path(f"storyboard/{session_id}")
        self.agents_dir = self.session_dir / "agents"

    def save_session_state(
        self,
        message: AgentMessage,
        request_data: Dict[str, Any],
        memory_history: list,
        iteration_count: int,
    ):
        """Save session state to disk following tech_specs.md structure"""
        try:
            # Create session directory structure
            self.agents_dir.mkdir(parents=True, exist_ok=True)

            # Save Brown's state
            brown_state = {
                "session_id": self.session_id,
                "conversation_id": self.conversation_id,
                "iteration_count": iteration_count,
                "memory": memory_history,
                "last_message": message.to_dict(),
                "original_request": request_data,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }

            with open(self.agents_dir / "brown_state.json", "w") as f:
                json.dump(brown_state, f, indent=2)

            # Save conversation log
            self._save_conversation_log(message)

            logger.info(f"Saved session state to {self.session_dir}")

        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    def _save_conversation_log(self, message: AgentMessage):
        """Save or update conversation log"""
        conversation_log = {
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "messages": [message.to_dict()],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        log_file = self.agents_dir / "conversation_log.json"
        if log_file.exists():
            # Append to existing log
            try:
                with open(log_file, "r") as f:
                    existing_log = json.load(f)
                existing_log["messages"].append(message.to_dict())
                existing_log["updated_at"] = conversation_log["updated_at"]
                conversation_log = existing_log
            except Exception as e:
                logger.warning(
                    f"Could not read existing log, creating new: {e}"
                )

        with open(log_file, "w") as f:
            json.dump(conversation_log, f, indent=2)

    def load_session_state(self) -> Dict[str, Any]:
        """Load session state from disk"""
        try:
            state_file = self.agents_dir / "brown_state.json"
            if state_file.exists():
                with open(state_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")

        return {}

    def session_exists(self) -> bool:
        """Check if session directory exists"""
        return self.session_dir.exists()

    def get_conversation_log(self) -> Dict[str, Any]:
        """Get conversation log"""
        try:
            log_file = self.agents_dir / "conversation_log.json"
            if log_file.exists():
                with open(log_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversation log: {e}")

        return {"messages": []}
