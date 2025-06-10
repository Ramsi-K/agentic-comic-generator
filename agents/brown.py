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

Core AgentBrown class with validation, processing, and review capabilities
"""

import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import os
import json
from pathlib import Path
from datetime import datetime

# LlamaIndex imports for multimodal ReActAgent
try:
    from llama_index.multi_modal_llms.openai import OpenAIMultiModal
    from llama_index.core.agent import ReActAgent
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.tools import FunctionTool, BaseTool
    from llama_index.core.llms import (
        ChatMessage,
        ImageBlock,
        TextBlock,
        MessageRole,
    )

    from llama_index.core.schema import ImageNode, Document
    from llama_index.core import SimpleDirectoryReader
    from typing import cast
except ImportError:
    OpenAIMultiModal = None
    ReActAgent = None
    ChatMemoryBuffer = None
    FunctionTool = None
    BaseTool = None
    ChatMessage = None
    ImageBlock = None
    TextBlock = None
    MessageRole = None
    ImageNode = None
    Document = None

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

    def __init__(
        self, max_iterations: int = 3, openai_api_key: Optional[str] = None
    ):
        self.max_iterations = max_iterations
        self.session_id = None
        self.conversation_id = None
        self.iteration_count = 0

        # Initialize LLM for prompt enhancement
        self.llm = None
        try:
            if OpenAIMultiModal:
                self.llm = OpenAIMultiModal(
                    model="gpt-4o",
                    api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                    temperature=0.7,
                    max_tokens=2048,
                )
                logger.info("✓ Initialized GPT-4V")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize LLM: {e}")

        # Core services
        self.moderator = ContentModerator()
        self.style_tagger = StyleTagger()
        self.evaluator = SimpleEvaluator()

        # Session services (initialized later)
        self.memory = None
        self.message_factory = None
        self.session_manager = None

        logger.info("Agent Brown initialized with core services")

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

    def _ensure_session(self) -> bool:
        """Ensure session services are initialized"""
        if not all([self.memory, self.message_factory, self.session_manager]):
            logger.warning("Session services not initialized")
            self._initialize_session()
        return True

    def _safe_memory_add(self, role: str, content: str) -> None:
        """Safely add message to memory if available"""
        if self.memory:
            self.memory.add_message(role, content)

    def process_request(self, request: StoryboardRequest) -> AgentMessage:
        """Process incoming user request and create message for Agent Bayko"""
        self._ensure_session()
        logger.info(f"Processing request for session {self.session_id}")

        # Log user request and state to memory
        self._safe_memory_add(
            "system",
            f"Starting new request with session_id: {self.session_id}",
        )
        self._safe_memory_add("user", f"Original prompt: {request.prompt}")
        self._safe_memory_add(
            "system",
            f"Request parameters: {json.dumps(asdict(request), indent=2)}",
        )

        # Step 1: Validate input
        validation = self.validate_input(request)
        self._safe_memory_add(
            "system",
            f"Validation result: {json.dumps(asdict(validation), indent=2)}",
        )

        if not validation.is_valid():
            self._safe_memory_add(
                "system", f"Validation failed: {validation.issues}"
            )
            return self.message_factory.create_error_message(
                validation.issues, validation.suggestions
            )

        # Step 2: Use LLM to enhance prompt and analyze style
        try:
            if self.llm:
                enhancement_prompt = f"""Enhance this comic story prompt for visual storytelling:
                Original: {request.prompt}
                Style preference: {request.style_preference or 'any'}
                Panels: {request.panels}
                
                Provide:
                1. Enhanced story description
                2. Visual style suggestions
                3. Mood and atmosphere
                4. Color palette recommendations"""

                self._safe_memory_add(
                    "system", f"Sending prompt to LLM:\n{enhancement_prompt}"
                )

                enhancement = self.llm.complete(
                    enhancement_prompt, image_documents=[]
                ).text
                self._safe_memory_add(
                    "assistant", f"LLM enhanced prompt:\n{enhancement}"
                )
            else:
                enhancement = request.prompt
                self._safe_memory_add(
                    "system", "No LLM available, using original prompt"
                )

        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            enhancement = request.prompt
            self._safe_memory_add("system", f"LLM enhancement failed: {e}")

        # Step 3: Analyze and tag style
        style_analysis = self.style_tagger.analyze_style(
            enhancement, request.style_preference
        )
        self._safe_memory_add(
            "system",
            f"Style analysis: {json.dumps(asdict(style_analysis), indent=2)}",
        )

        # Step 4: Create message for Bayko
        if not self.message_factory:
            self._initialize_session()
        # Provide an empty list or appropriate dialogues if not available
        message = self.message_factory.create_generation_request(
            enhanced_prompt=enhancement,
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
            dialogues=[],  # Add this argument as required by the method signature
        )

        # Log to memory and save state
        if self.memory:
            self.memory.add_message(
                "assistant",
                f"Created generation request for Bayko with {len(style_analysis.style_tags)} style tags",
            )
        if not self.session_manager:
            self._initialize_session()
        if self.session_manager and self.memory:
            self.session_manager.save_session_state(
                message,
                asdict(request),
                self.memory.get_history(),
                self.iteration_count,
            )

        logger.info(f"Generated request message {message.message_id}")
        return message

    def _safe_image_to_node(self, doc: Document) -> Optional[ImageNode]:
        """Safely convert document to ImageNode"""
        try:
            if hasattr(doc, "image") and doc.image:
                return ImageNode(text=doc.text or "", image=doc.image)
        except Exception as e:
            self._safe_memory_add(
                "system", f"Failed to convert image to node: {e}"
            )
        return None

    def _safe_memory_add(self, role: str, content: str) -> None:
        """Safely add message to memory if available"""
        self._ensure_session()
        if self.memory:
            self.memory.add_message(role, content)

    async def review_output(
        self,
        bayko_response: Dict[str, Any],
        original_request: StoryboardRequest,
    ) -> Optional[AgentMessage]:
        """Review Agent Bayko's output using GPT-4o for image analysis"""
        self._ensure_session()

        # Log review start
        self._safe_memory_add(
            "system",
            f"""Starting review with GPT-4o: {json.dumps({
                'prompt': original_request.prompt,
                'panels': len(bayko_response.get('panels', [])),
                'iteration': self.iteration_count + 1
            }, indent=2)}""",
        )

        try:
            if not self.llm:
                raise ValueError("GPT-4o LLM not initialized")

            if "panels" not in bayko_response:
                raise ValueError("No panels found in Bayko's response")

            # Get session content directory
            content_dir = Path(f"storyboard/{self.session_id}/content")
            if not content_dir.exists():
                raise ValueError(f"Content directory not found: {content_dir}")

            # Prepare image files for analysis
            image_files = []
            for panel in bayko_response["panels"]:
                panel_path = content_dir / f"panel_{panel['id']}.png"
                if panel_path.exists():
                    image_files.append(str(panel_path))
                else:
                    self._safe_memory_add(
                        "system",
                        f"Warning: Panel image not found: {panel_path}",
                    )

            if not image_files:
                raise ValueError("No panel images found for review")

            # Load images using SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_files=image_files)
            raw_docs = reader.load_data()

            # Convert documents to ImageNodes
            image_nodes = []
            for doc in raw_docs:
                if node := self._safe_image_to_node(doc):
                    image_nodes.append(node)

            if not image_nodes:
                raise ValueError("Failed to load any valid images for review")

            self._safe_memory_add(
                "system",
                f"Successfully loaded {len(image_nodes)} images for GPT-4o review",
            )

            # Construct detailed review prompt
            review_prompt = f"""As an expert art director, analyze these comic panels against the user's original request:

            ORIGINAL REQUEST: {original_request.prompt}
            STYLE PREFERENCE: {original_request.style_preference or 'Not specified'}
            REQUESTED PANELS: {original_request.panels}

            Analyze the following aspects:
            1. Story Accuracy:
               - Do the panels accurately depict the requested story?
               - Are the main story beats present?
               
            2. Visual Storytelling:
               - Is the panel flow clear and logical?
               - Does the sequence effectively convey the narrative?
               
            3. Style & Aesthetics:
               - Does it match any requested style preferences?
               - Is the artistic quality consistent?
               
            4. Technical Quality:
               - Are the images clear and well-composed?
               - Is there appropriate detail and contrast?

            Make ONE of these decisions:
            - APPROVE: If panels successfully tell the story and meet quality standards
            - REFINE: If specific improvements would enhance the result (list them)
            - REJECT: If fundamental issues require complete regeneration

            Provide a clear, actionable analysis focusing on how well these panels fulfill the USER'S ORIGINAL REQUEST."""

            # Get GPT-4o analysis
            analysis = self.llm.complete(
                prompt=review_prompt, image_documents=image_nodes
            ).text

            self._safe_memory_add("assistant", f"GPT-4o Analysis:\n{analysis}")

            # Parse decision from analysis
            decision = "refine"  # Default to refine
            if "APPROVE" in analysis.upper():
                decision = "approve"
            elif "REJECT" in analysis.upper():
                decision = "reject"

            # Create evaluation result
            evaluation = {
                "decision": decision,
                "reason": analysis,
                "confidence": 0.85,  # High confidence with GPT-4o
                "original_prompt": original_request.prompt,
                "analyzed_panels": len(image_nodes),
                "style_match": original_request.style_preference or "any",
            }

            self._safe_memory_add(
                "system",
                f"""GPT-4o review complete:\n{json.dumps({
                    'decision': decision,
                    'confidence': 0.85,
                    'analyzed_panels': len(image_nodes)
                }, indent=2)}""",
            )

        except Exception as e:
            logger.error(f"GPT-4o review failed: {str(e)}")
            self._safe_memory_add(
                "system",
                f"GPT-4o review failed, falling back to basic evaluator: {str(e)}",
            )
            # Fallback to basic evaluator
            evaluation = self.evaluator.evaluate(
                bayko_response, original_request.prompt
            )

        # Ensure message factory is available
        if not self.message_factory:
            self._initialize_session()

        # Create appropriate response message
        if evaluation["decision"] == "approve":
            return self.message_factory.create_approval_message(
                bayko_response, evaluation, self.iteration_count
            )
        elif evaluation["decision"] == "reject":
            return self.message_factory.create_rejection_message(
                bayko_response, evaluation, self.iteration_count
            )
        else:
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

    def _initialize_session(
        self,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ):
        """Initialize a new session with optional existing IDs or generate new ones"""
        if not self.session_manager:
            self.session_manager = SessionManager()

        if not session_id:
            session_id = str(uuid.uuid4())

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        self.session_id = session_id
        self.conversation_id = conversation_id
        # Initialize session-specific services
        self.memory = AgentMemory(self.session_id, "brown")
        self.message_factory = MessageFactory(
            self.session_id, self.conversation_id
        )
        self.session_manager = SessionManager(
            self.session_id, self.conversation_id
        )

        # Log initialization
        logger.info(
            f"🧠 Brown initialized memory for session {self.session_id}"
        )
        if self.memory:
            self.memory.add_message(
                "system", f"Session initialized with ID: {self.session_id}"
            )


# Example usage and testing
def main():
    """Example usage of Agent Brown"""
    # Create Brown instance
    brown = AgentBrown(max_iterations=3)

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
        # Return the review result
        return review_result

    # Show session info
    print(f"\nSession info: {brown.get_session_info()}")


if __name__ == "__main__":
    main()
