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

import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Core services
from services.unified_memory import AgentMemory
from services.simple_evaluator import SimpleEvaluator
from services.content_moderator import ContentModerator
from services.style_tagger import StyleTagger
from services.message_factory import MessageFactory, AgentMessage, MessageType
from services.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # Initialize core services
        self.moderator = ContentModerator()
        self.style_tagger = StyleTagger()
        self.evaluator = SimpleEvaluator()

        # Session-specific services (initialized when session starts)
        self.memory = None
        self.message_factory = None
        self.session_manager = None

        logger.info("Agent Brown initialized with real services")

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
        if self.memory:
            self.memory.add_message(
                "assistant",
                f"Validated input: {result.status.value} (confidence: {confidence:.2f})",
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
        # Initialize new session first
        self._initialize_session()

        logger.info(f"Processing request for session {self.session_id}")

        # Log user request to memory
        self.memory.add_message("user", request.prompt)

        # Step 1: Validate input
        validation = self.validate_input(request)
        if not validation.is_valid():
            return self.message_factory.create_error_message(
                validation.issues, validation.suggestions
            )

        # Step 2: Analyze and tag style
        style_analysis = self.style_tagger.analyze_style(
            request.prompt, request.style_preference
        )

        # Step 3: Create message for Bayko
        message = self.message_factory.create_generation_request(
            enhanced_prompt=style_analysis.enhanced_prompt,
            original_prompt=request.prompt,
            style_tags=style_analysis.style_tags,
            panels=request.panels,
            language=request.language,
            extras=request.extras or [],
            style_config={
                "primary_style": style_analysis.detected_style,
                "mood": style_analysis.mood,
                "color_palette": style_analysis.color_palette,
                "confidence": style_analysis.confidence,
            },
            validation_score=validation.confidence_score,
            iteration=self.iteration_count,
        )

        # Log to memory
        self.memory.add_message(
            "assistant", f"Created generation request for Bayko"
        )

        # Save session state
        self.session_manager.save_session_state(
            message,
            asdict(request),
            self.memory.get_history(),
            self.iteration_count,
        )

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

        # Use real evaluator
        print(f"🔍 Brown evaluating Bayko's output...")
        evaluation = self.evaluator.evaluate(
            bayko_response, original_request.prompt
        )

        # Log evaluation to memory
        if self.memory:
            self.memory.add_message(
                "assistant",
                f"Evaluation: {evaluation['decision']} - {evaluation['reason']}",
            )

        # Handle evaluation decision
        if evaluation["decision"] == "approve":
            print(f"✅ Brown approved: {evaluation['reason']}")
            return self.message_factory.create_approval_message(
                bayko_response, evaluation, self.iteration_count
            )

        elif evaluation["decision"] == "reject":
            print(f"❌ Brown rejected: {evaluation['reason']}")
            return self.message_factory.create_rejection_message(
                bayko_response, evaluation, self.iteration_count
            )

        else:  # refine
            print(f"🔄 Brown requesting refinement: {evaluation['reason']}")
            return self.message_factory.create_refinement_message(
                bayko_response, evaluation, self.iteration_count
            )

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        memory_size = 0
        if self.memory:
            try:
                memory_size = len(self.memory.get_history())
            except:
                memory_size = 0

        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "iteration_count": self.iteration_count,
            "memory_size": memory_size,
            "max_iterations": self.max_iterations,
        }

    def _initialize_session(self):
        """Initialize a new session with unique IDs and services"""
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        self.iteration_count = 1

        # Initialize session-specific services
        self.memory = AgentMemory(self.session_id, "brown")
        self.message_factory = MessageFactory(
            self.session_id, self.conversation_id
        )
        self.session_manager = SessionManager(
            self.session_id, self.conversation_id
        )

        print(f"🧠 Brown initialized memory for session {self.session_id}")


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
