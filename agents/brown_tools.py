"""
Brown Agent Tools for LlamaIndex Integration
Wraps existing AgentBrown methods as LlamaIndex FunctionTools for hackathon demo
"""

import json
from typing import Dict, Any, List
from llama_index.core.tools import FunctionTool
from agents.brown import AgentBrown, StoryboardRequest


class BrownTools:
    """Tool wrapper class for Agent Brown's methods"""

    def __init__(self, max_iterations: int = 3):
        # Use your original AgentBrown class directly
        self.brown = AgentBrown(max_iterations)
        self._current_request = None

    def validate_input_tool(
        self,
        prompt: str,
        style_preference: str = None,
        panels: int = 4,
        language: str = "english",
        extras: str = "[]",
    ) -> str:
        """Validate user input for comic generation. Returns validation status and feedback."""
        try:
            extras_list = json.loads(extras) if extras else []
        except:
            extras_list = []

        request = StoryboardRequest(
            prompt=prompt,
            style_preference=style_preference,
            panels=panels,
            language=language,
            extras=extras_list,
        )

        # Store for later use
        self._current_request = request

        result = self.brown.validate_input(request)

        return json.dumps(
            {
                "status": result.status.value,
                "is_valid": result.is_valid(),
                "issues": result.issues,
                "suggestions": result.suggestions,
                "confidence_score": result.confidence_score,
            }
        )

    def process_request_tool(
        self,
        prompt: str,
        style_preference: str = None,
        panels: int = 4,
        language: str = "english",
        extras: str = "[]",
    ) -> str:
        """Process validated request and create structured message for Agent Bayko."""
        try:
            extras_list = json.loads(extras) if extras else []
        except:
            extras_list = []

        request = StoryboardRequest(
            prompt=prompt,
            style_preference=style_preference,
            panels=panels,
            language=language,
            extras=extras_list,
        )

        # Store for later use
        self._current_request = request

        message = self.brown.process_request(request)

        return json.dumps(
            {
                "message_id": message.message_id,
                "session_id": self.brown.session_id,
                "enhanced_prompt": message.payload.get("prompt", ""),
                "style_tags": message.payload.get("style_tags", []),
                "panels": message.payload.get("panels", 4),
                "status": "ready_for_bayko",
            }
        )

    def simulate_bayko_generation(self, message_data: str) -> str:
        """Simulate Agent Bayko's content generation for demo purposes."""
        try:
            data = json.loads(message_data)
        except:
            data = {"panels": 4}

        panels_count = data.get("panels", 4)
        style_tags = data.get("style_tags", ["studio_ghibli", "soft_lighting"])
        enhanced_prompt = data.get("enhanced_prompt", "")
        original_prompt = data.get("original_prompt", "")

        # Simulate Bayko's response with realistic image URLs for multimodal analysis
        bayko_response = {
            "session_id": self.brown.session_id,
            "panels": [
                {
                    "id": i + 1,
                    "description": f"Panel {i+1}: {original_prompt} - {', '.join(style_tags)} style",
                    "image_path": f"storyboard/{self.brown.session_id}/content/panel_{i+1}.png",
                    "image_url": f"https://example.com/generated/panel_{i+1}.png",  # For multimodal analysis
                    "audio_path": (
                        f"panel_{i+1}.mp3"
                        if "narration" in str(data)
                        else None
                    ),
                    "subtitles_path": (
                        f"panel_{i+1}.vtt"
                        if "subtitles" in str(data)
                        else None
                    ),
                }
                for i in range(panels_count)
            ],
            "style_tags": style_tags,
            "metadata": {
                "generation_time": "45s",
                "total_panels": panels_count,
                "status": "completed",
                "enhanced_prompt": enhanced_prompt,
                "original_prompt": original_prompt,
            },
        }

        return json.dumps(bayko_response)

    def review_bayko_output_tool(
        self, bayko_response_json: str, original_prompt: str
    ) -> str:
        """Review Agent Bayko's output and determine if refinement is needed."""
        try:
            bayko_response = json.loads(bayko_response_json)
        except:
            # Fallback response for demo
            bayko_response = {
                "panels": [{"id": 1, "description": "Generated content"}],
                "style_tags": ["studio_ghibli"],
                "metadata": {"generation_time": "45s"},
            }

        # Use stored request or create new one
        request = self._current_request or StoryboardRequest(
            prompt=original_prompt
        )

        result = self.brown.review_output(bayko_response, request)

        if result:
            # Extract decision from the result
            payload = result.payload
            if "approved_content" in payload:
                decision = "APPROVED"
                reason = "Content meets quality standards"
            elif "feedback" in payload:
                decision = (
                    payload["feedback"].get("decision", "REFINE").upper()
                )
                reason = payload["feedback"].get(
                    "reason", "Quality assessment needed"
                )
            else:
                decision = "REFINE"
                reason = "Content needs improvement"
        else:
            decision = "APPROVED"
            reason = "Content meets quality standards"

        return json.dumps(
            {
                "decision": decision,
                "reason": reason,
                "iteration": self.brown.iteration_count,
                "max_iterations": self.brown.max_iterations,
                "final": decision in ["APPROVED", "REJECTED"],
            }
        )

    def get_session_info_tool(self) -> str:
        """Get current session information and processing state."""
        info = self.brown.get_session_info()
        return json.dumps(
            {
                "session_id": info.get("session_id"),
                "iteration_count": info.get("iteration_count", 0),
                "max_iterations": info.get("max_iterations", 3),
                "memory_size": info.get("memory_size", 0),
                "status": "active" if info.get("session_id") else "inactive",
            }
        )

    def create_llamaindex_tools(self) -> List[FunctionTool]:
        """Create LlamaIndex FunctionTools from Brown's methods"""
        return [
            FunctionTool.from_defaults(
                fn=self.validate_input_tool,
                name="validate_input",
                description="Validate user input for comic generation. MUST be called first for any user prompt. Returns validation status, issues, and suggestions.",
            ),
            FunctionTool.from_defaults(
                fn=self.process_request_tool,
                name="process_request",
                description="Process validated request and create structured message for Agent Bayko. Call after validation passes. Returns enhanced prompt and generation parameters.",
            ),
            FunctionTool.from_defaults(
                fn=self.simulate_bayko_generation,
                name="simulate_bayko_generation",
                description="Simulate Agent Bayko's content generation process. Takes processed request data and returns generated comic content.",
            ),
            FunctionTool.from_defaults(
                fn=self.review_bayko_output_tool,
                name="review_bayko_output",
                description="Review Agent Bayko's generated content and decide if refinement is needed. Returns approval, refinement request, or rejection decision.",
            ),
            FunctionTool.from_defaults(
                fn=self.get_session_info_tool,
                name="get_session_info",
                description="Get current session information and processing state. Use to track progress and iterations.",
            ),
        ]


def create_brown_tools(max_iterations: int = 3) -> BrownTools:
    """Factory function to create BrownTools instance"""
    return BrownTools(max_iterations=max_iterations)
