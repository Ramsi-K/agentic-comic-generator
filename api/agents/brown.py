"""
Agent Brown - The Orchestrator Agent

Agent Brown is the front-facing orchestrator that handles:
- Prompt validation and moderation
- Style tagging and enhancement
- JSON packaging for Agent Bayko
- Feedback review and refinement requests
- Session state management via LlamaIndex

This is the entry point for all user requests and manages the multi-turn
feedback loop with Agent Bayko for iterative comic generation.
"""

import json
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# TODO: Replace with actual LlamaIndex imports when available
# from llama_index.core.agent import ReActAgent
# from llama_index.core.memory import ChatMemoryBuffer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages Agent Brown can send"""

    GENERATION_REQUEST = "generation_request"
    REFINEMENT_REQUEST = "refinement_request"
    VALIDATION_ERROR = "validation_error"
    FINAL_APPROVAL = "final_approval"


class ValidationStatus(Enum):
    """Validation result statuses"""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Result of input validation"""

    status: ValidationStatus
    issues: List[str]
    suggestions: List[str]
    confidence_score: float

    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID


@dataclass
class StyleAnalysis:
    """Result of style analysis and tagging"""

    detected_style: str
    style_tags: List[str]
    mood: str
    color_palette: str
    enhanced_prompt: str
    confidence: float


@dataclass
class AgentMessage:
    """Schema for inter-agent communication following tech_specs.md"""

    message_id: str
    timestamp: str
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class StoryboardRequest:
    """Incoming request from user interface"""

    prompt: str
    style_preference: Optional[str] = None
    panels: int = 4
    language: str = "english"
    extras: Optional[List[str]] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = []


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
        for pattern in self.profanity_patterns:
            if re.search(pattern, text_lower):
                issues.append(
                    f"Content may contain inappropriate material: {pattern}"
                )

        # Check length
        if len(text.strip()) < 5:
            issues.append("Content too short to evaluate properly")

        return len(issues) == 0, issues


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


class FeedbackAnalyzer:
    """Analyzes generated content and provides feedback for refinement"""

    def __init__(self):
        self.quality_thresholds = {
            "minimum_score": 0.6,
            "good_score": 0.8,
            "excellent_score": 0.9,
        }

    def analyze_output(
        self,
        generated_content: Dict[str, Any],
        original_request: StoryboardRequest,
    ) -> Dict[str, Any]:
        """
        Analyze Bayko's output against the original request

        Args:
            generated_content: Content generated by Agent Bayko
            original_request: Original user request

        Returns:
            Feedback analysis with scores and recommendations
        """
        feedback = {
            "overall_score": 0.0,
            "adherence_score": 0.0,
            "style_consistency": 0.0,
            "narrative_flow": 0.0,
            "technical_quality": 0.0,
            "refinement_needed": False,
            "specific_feedback": [],
            "improvement_suggestions": [],
            "approved": False,
        }

        # Analyze different aspects (simplified scoring for now)
        # In production, this would use AI models for evaluation

        # Check if expected panels were generated
        if "panels" in generated_content:
            panel_count = len(generated_content.get("panels", []))
            expected_panels = original_request.panels

            if panel_count == expected_panels:
                feedback["adherence_score"] = 0.9
                feedback["specific_feedback"].append(
                    f"Generated {panel_count} panels as requested"
                )
            elif panel_count > 0:
                feedback["adherence_score"] = 0.6
                feedback["improvement_suggestions"].append(
                    f"Expected {expected_panels} panels, got {panel_count}"
                )
            else:
                feedback["adherence_score"] = 0.2
                feedback["improvement_suggestions"].append(
                    "No panels were generated"
                )

        # Check style consistency
        if "style_tags" in generated_content:
            feedback["style_consistency"] = 0.8
            feedback["specific_feedback"].append(
                "Style tags applied consistently"
            )
        else:
            feedback["style_consistency"] = 0.4
            feedback["improvement_suggestions"].append(
                "Apply consistent visual style across panels"
            )

        # Check narrative flow (simplified)
        if "narrative_structure" in generated_content:
            feedback["narrative_flow"] = 0.85
        else:
            feedback["narrative_flow"] = 0.6
            feedback["improvement_suggestions"].append(
                "Improve narrative flow between panels"
            )

        # Technical quality check
        if (
            "metadata" in generated_content
            and "errors" not in generated_content
        ):
            feedback["technical_quality"] = 0.9
        else:
            feedback["technical_quality"] = 0.5
            feedback["improvement_suggestions"].append(
                "Address technical issues in generation"
            )

        # Calculate overall score
        scores = [
            feedback["adherence_score"],
            feedback["style_consistency"],
            feedback["narrative_flow"],
            feedback["technical_quality"],
        ]
        feedback["overall_score"] = sum(scores) / len(scores)

        # Determine if refinement is needed
        feedback["refinement_needed"] = (
            feedback["overall_score"]
            < self.quality_thresholds["minimum_score"]
        )
        feedback["approved"] = (
            feedback["overall_score"] >= self.quality_thresholds["good_score"]
        )

        return feedback


class LlamaIndexMemoryStub:
    """
    Stub implementation of LlamaIndex memory for state management
    TODO: Replace with actual LlamaIndex ChatMemoryBuffer when available
    """

    def __init__(self, token_limit: int = 4000):
        self.token_limit = token_limit
        self.messages = []
        self.metadata = {}

    def add_message(
        self, message: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to memory"""
        self.messages.append(
            {
                "content": message,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            }
        )

        # Simple token limit enforcement (rough approximation)
        while (
            len(str(self.messages)) > self.token_limit * 4
        ):  # ~4 chars per token
            self.messages.pop(0)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages from memory"""
        return self.messages.copy()

    def clear(self):
        """Clear memory"""
        self.messages.clear()

    def get_context_string(self) -> str:
        """Get memory as context string"""
        return "\n".join([msg["content"] for msg in self.messages])


class AgentBrown:
    """
    Agent Brown - The Orchestrator

    Main responsibilities:
    - Validate and moderate user input
    - Analyze and tag visual styles
    - Package requests for Agent Bayko
    - Review generated content and provide feedback
    - Manage multi-turn refinement loops
    - Maintain session state and memory
    """

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.session_id = None
        self.conversation_id = None
        self.iteration_count = 0

        # Initialize components
        self.moderator = ContentModerator()
        self.style_tagger = StyleTagger()
        self.feedback_analyzer = FeedbackAnalyzer()

        # Initialize memory (stub implementation)
        self.memory = LlamaIndexMemoryStub(token_limit=4000)

        # TODO: Initialize actual LlamaIndex ReActAgent when available
        # self.agent = ReActAgent.from_tools(
        #     tools=[validation_tool, feedback_tool, assembly_tool],
        #     memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
        #     verbose=True
        # )

        logger.info("Agent Brown initialized")

    def validate_input(self, request: StoryboardRequest) -> ValidationResult:
        """
        Validate user input for appropriateness and completeness

        Args:
            request: User's storyboard request

        Returns:
            ValidationResult with status and feedback
        """
        issues = []
        suggestions = []

        # Basic validation
        if not request.prompt or len(request.prompt.strip()) < 10:
            issues.append(
                "Prompt too short - needs more detail for story generation"
            )
            suggestions.append(
                "Add more context about characters, setting, emotions, or plot"
            )

        if len(request.prompt) > 1000:
            issues.append("Prompt too long - may lose focus during generation")
            suggestions.append(
                "Condense to key story elements and main narrative arc"
            )

        # Content moderation
        is_safe, moderation_issues = self.moderator.check_content(
            request.prompt
        )
        if not is_safe:
            issues.extend(moderation_issues)
            suggestions.append(
                "Please revise content to ensure it's family-friendly"
            )

        # Panel count validation
        if request.panels < 1 or request.panels > 12:
            issues.append(
                f"Panel count ({request.panels}) outside recommended range (1-12)"
            )
            suggestions.append("Use 3-6 panels for optimal storytelling flow")

        # Language validation
        supported_languages = [
            "english",
            "korean",
            "japanese",
            "spanish",
            "french",
        ]
        if request.language.lower() not in supported_languages:
            issues.append(
                f"Language '{request.language}' may not be fully supported"
            )
            suggestions.append(
                f"Consider using: {', '.join(supported_languages)}"
            )

        # Calculate confidence score
        confidence = max(
            0.0, 1.0 - (len(issues) * 0.3) - (len(suggestions) * 0.1)
        )

        # Determine status
        if issues:
            status = ValidationStatus.INVALID
        elif suggestions:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        result = ValidationResult(
            status=status,
            issues=issues,
            suggestions=suggestions,
            confidence_score=confidence,
        )

        # Log validation to memory
        self.memory.add_message(
            f"Validated input: {result.status.value} (confidence: {confidence:.2f})",
            {"validation_result": asdict(result)},
        )

        return result

    def process_request(self, request: StoryboardRequest) -> AgentMessage:
        """
        Process incoming user request and create message for Agent Bayko

        Args:
            request: User's storyboard request

        Returns:
            AgentMessage formatted for Agent Bayko
        """
        # Initialize new session
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        self.iteration_count = 1

        logger.info(f"Processing request for session {self.session_id}")

        # Step 1: Validate input
        validation = self.validate_input(request)
        if not validation.is_valid():
            return self._create_error_message(
                validation.issues, validation.suggestions
            )

        # Step 2: Analyze and tag style
        style_analysis = self.style_tagger.analyze_style(
            request.prompt, request.style_preference
        )

        # Step 3: Create payload for Bayko
        payload = {
            "prompt": style_analysis.enhanced_prompt,
            "original_prompt": request.prompt,
            "style_tags": style_analysis.style_tags,
            "panels": request.panels,
            "language": request.language,
            "extras": request.extras,
            "style_config": {
                "primary_style": style_analysis.detected_style,
                "mood": style_analysis.mood,
                "color_palette": style_analysis.color_palette,
                "confidence": style_analysis.confidence,
            },
            "generation_params": {
                "quality": "high",
                "aspect_ratio": "16:9",
                "panel_layout": "sequential",
            },
        }

        # Step 4: Create agent message
        message = AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_brown",
            recipient="agent_bayko",
            message_type=MessageType.GENERATION_REQUEST.value,
            payload=payload,
            context={
                "conversation_id": self.conversation_id,
                "session_id": self.session_id,
                "iteration": self.iteration_count,
                "previous_feedback": None,
                "validation_score": validation.confidence_score,
            },
        )

        # Log to memory
        self.memory.add_message(
            f"Created generation request for Bayko",
            {
                "message": message.to_dict(),
                "style_analysis": asdict(style_analysis),
            },
        )

        # Save session state
        self._save_session_state(message, request)

        logger.info(f"Generated request message {message.message_id}")
        return message

    def review_output(
        self,
        bayko_response: Dict[str, Any],
        original_request: StoryboardRequest,
    ) -> Optional[AgentMessage]:
        """
        Review Agent Bayko's output and determine if refinement is needed

        Args:
            bayko_response: Generated content from Agent Bayko
            original_request: Original user request for context

        Returns:
            AgentMessage for refinement request, or None if approved
        """
        self.iteration_count += 1

        logger.info(
            f"Reviewing Bayko output (iteration {self.iteration_count})"
        )

        # Analyze the output
        feedback = self.feedback_analyzer.analyze_output(
            bayko_response, original_request
        )

        # Log feedback to memory
        self.memory.add_message(
            f"Analyzed Bayko output: score {feedback['overall_score']:.2f}",
            {"feedback": feedback},
        )

        # Check if approved or max iterations reached
        if feedback["approved"] or self.iteration_count > self.max_iterations:
            logger.info("Content approved or max iterations reached")
            return self._create_approval_message(bayko_response, feedback)

        # Create refinement request if needed
        if feedback["refinement_needed"]:
            logger.info("Refinement needed, creating refinement request")
            return self._create_refinement_message(bayko_response, feedback)

        # Content is acceptable but not excellent
        logger.info("Content acceptable, approving")
        return self._create_approval_message(bayko_response, feedback)

    def _create_error_message(
        self, issues: List[str], suggestions: List[str]
    ) -> AgentMessage:
        """Create error message for validation failures"""
        return AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_brown",
            recipient="user_interface",
            message_type=MessageType.VALIDATION_ERROR.value,
            payload={
                "error": "Input validation failed",
                "issues": issues,
                "suggestions": suggestions,
            },
            context={
                "conversation_id": self.conversation_id or "error",
                "session_id": self.session_id or "error",
                "iteration": 0,
                "error_type": "validation",
            },
        )

    def _create_refinement_message(
        self, bayko_response: Dict[str, Any], feedback: Dict[str, Any]
    ) -> AgentMessage:
        """Create refinement request message"""
        return AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_brown",
            recipient="agent_bayko",
            message_type=MessageType.REFINEMENT_REQUEST.value,
            payload={
                "original_content": bayko_response,
                "feedback": feedback,
                "specific_improvements": feedback["improvement_suggestions"],
                "focus_areas": [
                    area
                    for area, score in [
                        ("adherence", feedback["adherence_score"]),
                        ("style_consistency", feedback["style_consistency"]),
                        ("narrative_flow", feedback["narrative_flow"]),
                        ("technical_quality", feedback["technical_quality"]),
                    ]
                    if score < 0.7
                ],
                "iteration": self.iteration_count,
            },
            context={
                "conversation_id": self.conversation_id,
                "session_id": self.session_id,
                "iteration": self.iteration_count,
                "previous_feedback": feedback,
                "refinement_reason": "Quality below threshold",
            },
        )

    def _create_approval_message(
        self, bayko_response: Dict[str, Any], feedback: Dict[str, Any]
    ) -> AgentMessage:
        """Create final approval message"""
        return AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_brown",
            recipient="user_interface",
            message_type=MessageType.FINAL_APPROVAL.value,
            payload={
                "approved_content": bayko_response,
                "final_feedback": feedback,
                "session_summary": {
                    "total_iterations": self.iteration_count,
                    "final_score": feedback["overall_score"],
                    "processing_complete": True,
                },
            },
            context={
                "conversation_id": self.conversation_id,
                "session_id": self.session_id,
                "iteration": self.iteration_count,
                "final_approval": True,
                "completion_timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    def _save_session_state(
        self, message: AgentMessage, request: StoryboardRequest
    ):
        """Save session state to disk following tech_specs.md structure"""
        try:
            # Create session directory structure
            session_dir = Path(f"storyboard/{self.session_id}")
            agents_dir = session_dir / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)

            # Save Brown's state
            brown_state = {
                "session_id": self.session_id,
                "conversation_id": self.conversation_id,
                "iteration_count": self.iteration_count,
                "memory": self.memory.get_messages(),
                "last_message": message.to_dict(),
                "original_request": asdict(request),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }

            with open(agents_dir / "brown_state.json", "w") as f:
                json.dump(brown_state, f, indent=2)

            # Save conversation log
            conversation_log = {
                "conversation_id": self.conversation_id,
                "session_id": self.session_id,
                "messages": [message.to_dict()],
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }

            log_file = agents_dir / "conversation_log.json"
            if log_file.exists():
                # Append to existing log
                with open(log_file, "r") as f:
                    existing_log = json.load(f)
                existing_log["messages"].append(message.to_dict())
                existing_log["updated_at"] = conversation_log["updated_at"]
                conversation_log = existing_log

            with open(log_file, "w") as f:
                json.dump(conversation_log, f, indent=2)

            logger.info(f"Saved session state to {session_dir}")

        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "iteration_count": self.iteration_count,
            "memory_size": len(self.memory.get_messages()),
            "max_iterations": self.max_iterations,
        }


# Factory function for easy instantiation
def create_agent_brown(max_iterations: int = 3) -> AgentBrown:
    """
    Create and configure Agent Brown instance

    Args:
        max_iterations: Maximum number of refinement iterations

    Returns:
        Configured AgentBrown instance
    """
    return AgentBrown(max_iterations=max_iterations)


# Example usage and testing
def main():
    """Example usage of Agent Brown"""
    # Create agent
    brown = create_agent_brown()

    # Example request
    request = StoryboardRequest(
        prompt="A moody K-pop idol finds a puppy on the street. "
        "It changes everything.",
        style_preference="studio_ghibli",
        panels=4,
        language="korean",
        extras=["narration", "subtitles"],
    )

    # Process request
    message = brown.process_request(request)
    print("Generated message for Bayko:")
    print(message.to_json())

    # Example Bayko response (simulated)
    bayko_response = {
        "panels": [
            {"id": 1, "description": "Idol walking alone"},
            {"id": 2, "description": "Discovers puppy"},
            {"id": 3, "description": "Moment of connection"},
            {"id": 4, "description": "Walking together"},
        ],
        "style_tags": ["whimsical", "soft_lighting"],
        "metadata": {"generation_time": "45s"},
    }

    # Review output
    review_result = brown.review_output(bayko_response, request)
    if review_result:
        print("\nReview result:")
        print(review_result.to_json())

    # Show session info
    print(f"\nSession info: {brown.get_session_info()}")


if __name__ == "__main__":
    main()
