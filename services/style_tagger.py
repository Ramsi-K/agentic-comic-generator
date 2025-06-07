import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


@dataclass
class StyleAnalysis:
    """Result of style analysis and tagging"""

    detected_style: str
    style_tags: List[str]
    mood: str
    color_palette: str
    enhanced_prompt: str
    confidence: float


class StyleTagger:
    """Handles style detection and tagging"""

    def __init__(self):
        self.style_mappings = {
            "studio_ghibli": {
                "keywords": [
                    "ghibli",
                    "whimsical",
                    "nature",
                    "peaceful",
                    "magical",
                ],
                "tags": ["whimsical", "nature", "soft_lighting", "watercolor"],
                "mood": "peaceful",
                "color_palette": "warm_earth_tones",
            },
            "manga": {
                "keywords": [
                    "manga",
                    "anime",
                    "dramatic",
                    "action",
                    "intense",
                ],
                "tags": [
                    "high_contrast",
                    "dramatic",
                    "speed_lines",
                    "screentones",
                ],
                "mood": "dynamic",
                "color_palette": "black_white_accent",
            },
            "western": {
                "keywords": ["superhero", "comic", "bold", "heroic", "action"],
                "tags": [
                    "bold_lines",
                    "primary_colors",
                    "superhero",
                    "action",
                ],
                "mood": "heroic",
                "color_palette": "bright_primary",
            },
            "whisper_soft": {
                "keywords": ["soft", "gentle", "quiet", "subtle", "whisper"],
                "tags": ["pastel", "dreamy", "soft_focus", "ethereal"],
                "mood": "contemplative",
                "color_palette": "muted_pastels",
            },
        }

        self.mood_keywords = {
            "peaceful": ["calm", "serene", "gentle", "quiet", "peaceful"],
            "dramatic": [
                "intense",
                "conflict",
                "tension",
                "dramatic",
                "powerful",
            ],
            "whimsical": [
                "magical",
                "wonder",
                "fantasy",
                "dream",
                "whimsical",
            ],
            "melancholy": [
                "sad",
                "lonely",
                "lost",
                "melancholy",
                "bittersweet",
            ],
            "energetic": [
                "action",
                "fast",
                "dynamic",
                "energetic",
                "exciting",
            ],
        }

    def analyze_style(
        self, prompt: str, style_preference: Optional[str] = None
    ) -> StyleAnalysis:
        """
        Analyze prompt and determine appropriate style tags

        Args:
            prompt: User's story prompt
            style_preference: Optional user-specified style preference

        Returns:
            StyleAnalysis with detected style and tags
        """
        prompt_lower = prompt.lower()

        # If user specified a style, use it if valid
        if (
            style_preference
            and style_preference.lower() in self.style_mappings
        ):
            style_key = style_preference.lower()
            confidence = 0.9  # High confidence for user-specified style
        else:
            # Auto-detect style based on keywords
            style_scores = {}
            for style, config in self.style_mappings.items():
                score = sum(
                    1
                    for keyword in config["keywords"]
                    if keyword in prompt_lower
                )
                if score > 0:
                    style_scores[style] = score

            if style_scores:
                style_key = max(
                    style_scores.keys(), key=lambda x: style_scores[x]
                )
                confidence = min(0.8, style_scores[style_key] * 0.2)
            else:
                # Default to studio_ghibli for unknown styles
                style_key = "studio_ghibli"
                confidence = 0.5

        style_config = self.style_mappings[style_key]

        # Detect mood
        detected_moods = []
        for mood, keywords in self.mood_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_moods.append(mood)

        # Use style's default mood if none detected
        final_mood = (
            detected_moods[0] if detected_moods else style_config["mood"]
        )

        # Enhance prompt with style information
        enhanced_prompt = self._enhance_prompt(
            prompt, style_config, final_mood
        )

        return StyleAnalysis(
            detected_style=style_key,
            style_tags=style_config["tags"],
            mood=final_mood,
            color_palette=style_config["color_palette"],
            enhanced_prompt=enhanced_prompt,
            confidence=confidence,
        )

    def _enhance_prompt(
        self, original_prompt: str, style_config: Dict, mood: str
    ) -> str:
        """Enhance the original prompt with style-specific details"""
        style_descriptors = ", ".join(style_config["tags"])
        return f"{original_prompt}. Visual style: {style_descriptors}, mood: {mood}, color palette: {style_config['color_palette']}"
