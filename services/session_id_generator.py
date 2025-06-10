"""
Session ID Generator Service
Provides consistent session ID generation across the application.
"""

import re
import uuid
from datetime import datetime
from typing import Optional


class SessionIdGenerator:
    """Generates consistent session IDs across the application"""

    @staticmethod
    def create_session_id(prefix: Optional[str] = None) -> str:
        """
        Create a new session ID.
        For production: Uses UUID format
        For testing: Uses sequential format with optional prefix
        """
        if prefix:
            # Use consistent format for test sessions
            session_num = uuid.uuid4().int % 1000  # Get last 3 digits
            return f"{prefix}_{session_num:03d}"

        # Production session ID - UUID based
        return str(uuid.uuid4())

    @staticmethod
    def is_valid_session_id(session_id: str) -> bool:
        """Validate if a session ID follows the expected format"""
        # UUID format (production)
        uuid_pattern = (
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        # Test format (test_001, etc)
        test_pattern = r"^[a-z]+_\d{3}$"

        return bool(
            re.match(uuid_pattern, session_id, re.I)
            or re.match(test_pattern, session_id)
        )
