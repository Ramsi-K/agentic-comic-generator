import re
from typing import Tuple, List


class ContentModerator:
    """Handles content moderation and profanity filtering"""

    def __init__(self):
        # Basic profanity patterns - in production, use a proper filter
        self.profanity_patterns = [
            r"\b(explicit|inappropriate|offensive)\b",
            r"\b(violence|gore|blood)\b",
            r"\b(hate|discrimination|bias)\b",
            r"\b(nsfw|adult|sexual)\b",
        ]

        # Content safety keywords
        self.safety_keywords = [
            "safe",
            "family-friendly",
            "appropriate",
            "wholesome",
        ]

    def check_content(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check content for appropriateness

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []
        text_lower = text.lower()

        # Check for profanity patterns
        issues.extend(
            [
                f"Content may contain inappropriate material: {pattern}"
                for pattern in self.profanity_patterns
                if re.search(pattern, text_lower)
            ]
        )

        # Check length
        if len(text.strip()) < 5:
            issues.append("Content too short to evaluate properly")

        return len(issues) == 0, issues
