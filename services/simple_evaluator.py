class SimpleEvaluator:
    """Basic evaluation logic for Brown's decision making"""

    MAX_ATTEMPTS = 3  # Original + 2 revisions

    def __init__(self):
        self.attempt_count = 0

    def evaluate(self, bayko_output: dict, original_prompt: str) -> dict:
        """Evaluate Bayko's output and decide: approve, reject, or refine"""
        self.attempt_count += 1

        print(
            f"🔍 Brown evaluating attempt {self.attempt_count}/{self.MAX_ATTEMPTS}"
        )

        # Rule 1: Auto-reject if dialogue in images
        if self._has_dialogue_in_images(bayko_output):
            return {
                "decision": "reject",
                "reason": "Images contain dialogue text - use subtitles instead",
                "final": True,
            }

        # Rule 2: Auto-reject if story is incoherent
        if not self._is_story_coherent(bayko_output):
            return {
                "decision": "reject",
                "reason": "Story panels don't follow logical sequence",
                "final": True,
            }

        # Rule 3: Force approve if max attempts reached
        if self.attempt_count >= self.MAX_ATTEMPTS:
            return {
                "decision": "approve",
                "reason": f"Max attempts ({self.MAX_ATTEMPTS}) reached - accepting current quality",
                "final": True,
            }

        # Rule 4: Check if output matches prompt intent
        if self._matches_prompt_intent(bayko_output, original_prompt):
            return {
                "decision": "approve",
                "reason": "Output matches prompt and quality is acceptable",
                "final": True,
            }
        else:
            return {
                "decision": "refine",
                "reason": "Output needs improvement to better match prompt",
                "final": False,
            }

    def _has_dialogue_in_images(self, output: dict) -> bool:
        """Check if panels mention dialogue in the image"""
        panels = output.get("panels", [])

        dialogue_keywords = [
            "speech bubble",
            "dialogue",
            "talking",
            "saying",
            "text in image",
            "speech",
            "conversation",
        ]

        for panel in panels:
            description = panel.get("description", "").lower()
            if any(keyword in description for keyword in dialogue_keywords):
                print(f"❌ Found dialogue in image: {description}")
                return True

        return False

    def _is_story_coherent(self, output: dict) -> bool:
        """Basic check for story coherence"""
        panels = output.get("panels", [])

        if len(panels) < 2:
            return True  # Single panel is always coherent

        # Check 1: All panels should have descriptions
        descriptions = [p.get("description", "") for p in panels]
        if any(not desc.strip() for desc in descriptions):
            print("❌ Some panels missing descriptions")
            return False

        # Check 2: Panels shouldn't be identical (no progression)
        if len(set(descriptions)) == 1:
            print("❌ All panels are identical - no story progression")
            return False

        # Check 3: Look for obvious incoherence keywords
        incoherent_keywords = [
            "unrelated",
            "random",
            "doesn't make sense",
            "no connection",
            "contradictory",
        ]

        full_text = " ".join(descriptions).lower()
        if any(keyword in full_text for keyword in incoherent_keywords):
            print("❌ Story contains incoherent elements")
            return False

        return True

    def _matches_prompt_intent(self, output: dict, prompt: str) -> bool:
        """Check if output generally matches the original prompt"""
        panels = output.get("panels", [])

        if not panels:
            return False

        # Simple keyword matching
        prompt_words = set(prompt.lower().split())
        panel_text = " ".join(
            [p.get("description", "") for p in panels]
        ).lower()
        panel_words = set(panel_text.split())

        # At least 20% of prompt words should appear in panel descriptions
        overlap = len(prompt_words.intersection(panel_words))
        match_ratio = overlap / len(prompt_words) if prompt_words else 0

        print(f"📊 Prompt match ratio: {match_ratio:.2f}")
        return match_ratio >= 0.2

    def reset(self):
        """Reset for new session"""
        self.attempt_count = 0
