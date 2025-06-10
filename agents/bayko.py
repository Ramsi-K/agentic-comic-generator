"""
Agent Bayko - The Creative Engine

Agent Bayko is the content generation specialist that handles:
- Reading structured JSON requests from Agent Brown
- Generating comic panel images using SDXL via Modal compute
- Creating audio narration using TTS services
- Generating subtitle files in VTT format
- Managing output files in the session directory structure
- Updating metadata as content is generated
- Supporting refinement requests from Brown's feedback loop
"""

import uuid
import json
import time
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, cast
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum

from openai import OpenAI
from openai.types.chat import ChatCompletion
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# Core services
from services.unified_memory import AgentMemory
from services.session_manager import SessionManager as ServiceSessionManager
from services.message_factory import MessageFactory, AgentMessage, MessageType

# Tools
from agents.bayko_tools import ModalImageGenerator, ModalCodeExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationStatus(Enum):
    """Status of content generation"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REFINEMENT_NEEDED = "refinement_needed"


class ContentType(Enum):
    """Types of content Bayko can generate"""

    IMAGE = "image"
    AUDIO = "audio"
    SUBTITLES = "subtitles"
    METADATA = "metadata"


@dataclass
class PanelContent:
    """Content and metadata for a single comic panel"""

    panel_id: int
    description: str
    enhanced_prompt: str = ""  # LLM-enhanced prompt
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    audio_path: Optional[str] = None
    subtitles_path: Optional[str] = None
    status: str = "pending"
    generation_time: float = 0.0
    style_tags: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    refinement_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Result of a generation or refinement request"""

    session_id: str
    panels: List[PanelContent]
    metadata: Dict[str, Any]
    status: GenerationStatus
    total_time: float
    errors: List[str] = field(default_factory=list)
    refinement_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "panels": [panel.to_dict() for panel in self.panels],
            "metadata": self.metadata,
            "status": self.status.value,
            "total_time": self.total_time,
            "errors": self.errors,
            "refinement_applied": self.refinement_applied,
        }


class AgentBayko:
    """
    Agent Bayko - The Creative Engine

    Main responsibilities:
    - Process structured requests from Agent Brown
    - Generate comic panel images via SDXL/Modal
    - Create audio narration using TTS
    - Generate subtitle files in VTT format
    - Manage session file organization
    - Support refinement requests and feedback loops
    - Update metadata and performance metrics
    """

    def __init__(self, llm: Optional[LlamaOpenAI] = None):
        # Core tools
        self.image_generator = ModalImageGenerator()
        self.code_executor = ModalCodeExecutor()
        self.llm = llm

        # Session state
        self.current_session: Optional[str] = None
        self.session_manager: Optional[ServiceSessionManager] = None
        self.memory: Optional[AgentMemory] = None
        self.message_factory: Optional[MessageFactory] = None

        # Stats tracking
        self.generation_stats = {
            "panels_generated": 0,
            "refinements_applied": 0,
            "total_time": 0.0,
            "errors": [],
        }

    async def process_generation_request(
        self, message: Dict[str, Any]
    ) -> AgentMessage:
        """
        Process generation request from Agent Brown

        Args:
            message: AgentMessage from Brown containing generation request

        Returns:
            AgentMessage with generated content and metadata
        """
        start_time = time.time()

        # Extract request data
        payload = message.get("payload", {})
        context = message.get("context", {})
        session_id = context.get("session_id")

        if not session_id:
            raise ValueError("No session_id provided in message context")

        # Initialize session
        self._initialize_session(session_id, context.get("conversation_id"))
        logger.info(f"Processing generation request for session {session_id}")

        # Extract generation parameters
        prompt = payload.get("prompt", "")
        original_prompt = payload.get("original_prompt", "")
        style_tags = payload.get("style_tags", [])
        panel_count = payload.get("panels", 4)

        # Create panel descriptions (using Brown's enhanced prompt for now)
        panel_prompts = self._create_panel_descriptions(
            prompt, panel_count, payload.get("style_config", {})
        )

        # Generate content for each panel in parallel
        panels = []
        errors = []

        for i, description in enumerate(panel_prompts, 1):
            try:
                panel = await self._generate_panel_content(
                    panel_id=i,
                    description=description,
                    enhanced_prompt=prompt,  # Using Brown's enhanced prompt directly
                    style_tags=style_tags,
                    language="english",  # Default for now
                    extras=[],  # Simplified for now
                    session_id=session_id,
                )
                panels.append(panel)
                self._update_generation_progress(i, panel_count)

            except Exception as e:
                error_msg = f"Failed to generate panel {i}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                panels.append(
                    PanelContent(
                        panel_id=i, description=description, errors=[error_msg]
                    )
                )

        # Calculate total time
        total_time = time.time() - start_time

        # Create metadata
        metadata = self._create_generation_metadata(
            payload, panels, total_time, errors
        )

        # Update session metadata
        self.update_metadata(metadata)
        self._save_current_state(message, panels, metadata)

        # Create result object
        result = GenerationResult(
            session_id=session_id,
            panels=panels,
            metadata=metadata,
            status=(
                GenerationStatus.COMPLETED
                if not errors
                else GenerationStatus.FAILED
            ),
            total_time=total_time,
            errors=errors,
        )

        # Create response message
        if self.memory:
            self.memory.add_message(
                "assistant",
                f"Generated {len(panels)} panels in {total_time:.2f}s",
            )

        if self.message_factory:
            return self.message_factory.create_approval_message(
                result.to_dict(),
                {
                    "overall_score": 1.0 if not errors else 0.7,
                    "generation_successful": True,
                    "panels_generated": len(panels),
                    "total_time": total_time,
                },
                1,  # Initial iteration
            )

        # Create a plain message if no factory available
        plain_message = AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_bayko",
            recipient="agent_brown",
            message_type="generation_response",
            payload=result.to_dict(),
            context=context,
        )

        return plain_message

    async def process_refinement_request(
        self, message: Dict[str, Any]
    ) -> AgentMessage:
        """Process refinement request from Agent Brown"""

        start_time = time.time()

        # Extract request data
        payload = message.get("payload", {})
        context = message.get("context", {})
        session_id = context.get("session_id")
        iteration = context.get("iteration", 1)

        if not session_id:
            raise ValueError("No session_id provided in message context")

        # Initialize session if needed
        if self.current_session != session_id:
            self._initialize_session(
                session_id, context.get("conversation_id")
            )

        # Extract refinement data
        original_content = payload.get("original_content", {})
        feedback = payload.get("feedback", {})
        focus_areas = payload.get("focus_areas", [])
        refinements = payload.get("specific_improvements", [])

        # Log refinement request
        if self.memory:
            self.memory.add_message(
                "user",
                f"Refinement request received - Areas: {', '.join(focus_areas)}",
            )

        # Generate new panels for ones needing refinement
        panels = []
        errors = []

        # Convert original panels to PanelContent objects
        original_panels = [
            PanelContent(
                panel_id=p.get("panel_id"),
                description=p.get("description", ""),
                enhanced_prompt=p.get("enhanced_prompt", ""),
                image_path=p.get("image_path"),
                image_url=p.get("image_url"),
                style_tags=p.get("style_tags", []),
                status=p.get("status", "pending"),
                generation_time=p.get("generation_time", 0.0),
                errors=p.get("errors", []),
            )
            for p in original_content.get("panels", [])
        ]

        # Process each panel
        for panel in original_panels:
            try:
                if any(
                    area in focus_areas
                    for area in ["visual_quality", "style_consistency"]
                ):
                    # Step 1: Improve prompt based on feedback
                    improved_prompt = self._improve_prompt_with_feedback(
                        original_prompt=panel.enhanced_prompt,
                        feedback=feedback,
                        improvements=refinements,
                    )

                    # Step 2: Generate new panel with improved prompt
                    refined_panel = await self._generate_panel_content(
                        panel_id=panel.panel_id,
                        description=panel.description,
                        enhanced_prompt=improved_prompt,  # Use improved prompt
                        style_tags=panel.style_tags,
                        language="english",
                        extras=[],
                        session_id=session_id,
                    )

                    # Add refinement history
                    refined_panel.refinement_history.append(
                        {
                            "iteration": iteration,
                            "feedback": feedback,
                            "improvements": refinements,
                            "original_prompt": panel.enhanced_prompt,
                            "refined_prompt": improved_prompt,
                        }
                    )
                    panels.append(refined_panel)
                else:
                    # Keep original panel
                    panels.append(panel)

            except Exception as e:
                error_msg = (
                    f"Failed to refine panel {panel.panel_id}: {str(e)}"
                )
                logger.error(error_msg)
                errors.append(error_msg)
                panels.append(panel)  # Keep original on error

        total_time = time.time() - start_time

        # Create metadata
        metadata = {
            "refinement": {
                "iteration": iteration,
                "feedback": feedback,
                "improvements": refinements,
                "focus_areas": focus_areas,
                "panels_refined": len(
                    [p for p in panels if len(p.refinement_history) > 0]
                ),
                "total_time": total_time,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Save state
        self._save_current_state(message, panels, metadata)

        # Create result
        result = GenerationResult(
            session_id=session_id,
            panels=panels,
            metadata=metadata,
            status=(
                GenerationStatus.COMPLETED
                if not errors
                else GenerationStatus.FAILED
            ),
            total_time=total_time,
            errors=errors,
            refinement_applied=True,
        )

        # Log completion
        if self.memory:
            self.memory.add_message(
                "assistant",
                f"Refined {len([p for p in panels if len(p.refinement_history) > 0])} panels in {total_time:.2f}s",
            )

        # Create response message
        if self.message_factory:
            return self.message_factory.create_approval_message(
                result.to_dict(),
                {
                    "overall_score": 1.0 if not errors else 0.7,
                    "refinement_successful": True,
                    "panels_refined": len(
                        [p for p in panels if len(p.refinement_history) > 0]
                    ),
                    "total_time": total_time,
                    "iteration": iteration,
                },
                iteration,
            )

        # Fallback to direct response if no message factory
        return AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_bayko",
            recipient="agent_brown",
            message_type="refinement_response",
            payload=result.to_dict(),
            context=context,
        )

    def _create_panel_descriptions(
        self, prompt: str, panel_count: int, style_config: Dict[str, Any]
    ) -> List[str]:
        """
        Use mini LLM to break down the story prompt into panel descriptions

        Args:
            prompt: The main story prompt
            panel_count: Number of panels to generate
            style_config: Style configuration for the panels

        Returns:
            List of panel descriptions
        """
        if not self.llm:
            # Fallback to basic panel descriptions if LLM is not initialized
            logger.warning("LLM not available, using basic panel descriptions")
            return [
                f"{prompt} (Panel {i+1} of {panel_count})"
                for i in range(panel_count)
            ]

        try:
            system_prompt = f"""You are a comic storyboarding expert. Break down this story into {panel_count} engaging and visually interesting sequential panels.
            For each panel:
            - Focus on key narrative moments
            - Include visual composition guidance
            - Maintain continuity between panels
            - Consider dramatic timing and pacing
            DO NOT include panel numbers or "Panel X:" prefixes.
            Return ONLY the panel descriptions, one per line.
            Style notes: {json.dumps(style_config, indent=2)}"""

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            # Extract panel descriptions
            descriptions = []
            if response.choices[0].message.content:
                # Split on newlines and filter out empty lines
                descriptions = [
                    d.strip()
                    for d in response.choices[0].message.content.split("\n")
                    if d.strip()
                ]

            # Ensure we have exactly panel_count descriptions
            if len(descriptions) < panel_count:
                # Pad with basic descriptions if needed
                descriptions.extend(
                    [
                        f"{prompt} (Panel {i+1})"
                        for i in range(len(descriptions), panel_count)
                    ]
                )
            elif len(descriptions) > panel_count:
                # Trim extra descriptions
                descriptions = descriptions[:panel_count]

            return descriptions

        except Exception as e:
            logger.error(f"Failed to create panel descriptions: {e}")
            # Fallback to basic descriptions
            return [
                f"{prompt} (Panel {i+1} of {panel_count})"
                for i in range(panel_count)
            ]

    def generate_prompt_from_description(
        self, description: str, style_tags: List[str]
    ) -> str:
        """
        Use mini LLM to convert panel descriptions into SDXL-optimized prompts

        Args:
            description: The panel description to optimize
            style_tags: List of style tags to incorporate

        Returns:
            SDXL-optimized prompt
        """
        if not self.llm:
            # Fallback to basic prompt if LLM is not initialized
            style_str = ", ".join(style_tags) if style_tags else ""
            return f"{description} {style_str}".strip()

        try:
            system_prompt = """You are an expert at crafting prompts for SDXL image generation.
            Convert the given panel description into an optimized SDXL prompt that will generate a high-quality comic panel.
            Follow these guidelines:
            - Be specific about visual elements and composition
            - Maintain artistic consistency with the provided style
            - Use clear, direct language that SDXL will understand
            - Focus on key details that drive the narrative
            Return ONLY the optimized prompt text with no additional formatting."""

            style_context = ""
            if style_tags:
                style_context = (
                    f"\nStyle requirements: {', '.join(style_tags)}"
                )

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"{description}{style_context}",
                    },
                ],
                temperature=0.5,
                max_tokens=500,
            )

            if response.choices[0].message.content:
                return response.choices[0].message.content.strip()

            # Fallback if no valid response
            style_str = ", ".join(style_tags) if style_tags else ""
            return f"{description} {style_str}".strip()

        except Exception as e:
            logger.error(f"Failed to optimize prompt: {e}")
            # Fallback to basic prompt
            style_str = ", ".join(style_tags) if style_tags else ""
            return f"{description} {style_str}".strip()

    def _update_generation_progress(
        self, current_panel: int, total_panels: int
    ) -> None:
        """Update generation progress tracking"""
        progress = (current_panel / total_panels) * 100
        logger.info(
            f"Generation Progress: {progress:.1f}% ({current_panel}/{total_panels})"
        )

        self.generation_stats["panels_generated"] += 1

    async def _generate_panel_content(
        self,
        panel_id: int,
        description: str,
        enhanced_prompt: str,
        style_tags: List[str],
        language: str = "english",
        extras: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> PanelContent:
        """
        Generate content for a single panel

        Args:
            panel_id: Panel identifier
            description: Raw panel description
            enhanced_prompt: Initial enhanced prompt
            style_tags: Style tags to apply
            language: Language for text (default: english)
            extras: Additional generation parameters
            session_id: Current session ID

        Returns:
            PanelContent object with generated content
        """
        start_time = time.time()
        extras = extras or []

        try:
            # Step 1: Optimize prompt for SDXL using LLM
            optimized_prompt = self.generate_prompt_from_description(
                description=description, style_tags=style_tags
            )
            # Step 2: Generate image using optimized prompt
            generated_path, gen_time = (
                await self.image_generator.generate_panel_image(
                    prompt=optimized_prompt,
                    style_tags=style_tags,
                    panel_id=panel_id,
                    session_id=session_id,
                )
            )

            # Step 3: Create panel content object
            panel = PanelContent(
                panel_id=panel_id,
                description=description,
                enhanced_prompt=optimized_prompt,  # Store the optimized prompt
                image_path=generated_path,
                image_url=(
                    f"file://{generated_path}" if generated_path else None
                ),
                style_tags=style_tags,
                status=GenerationStatus.COMPLETED.value,
                generation_time=gen_time,
            )

            return panel

        except Exception as e:
            error_msg = f"Panel {panel_id} generation failed: {str(e)}"
            logger.error(error_msg)

            return PanelContent(
                panel_id=panel_id,
                description=description,
                enhanced_prompt=enhanced_prompt,
                style_tags=style_tags,
                status=GenerationStatus.FAILED.value,
                generation_time=time.time() - start_time,
                errors=[error_msg],
            )

    def _create_generation_metadata(
        self,
        payload: Dict[str, Any],
        panels: List[PanelContent],
        total_time: float,
        errors: List[str],
    ) -> Dict[str, Any]:
        """Create metadata for generation request"""
        return {
            "request": {
                "prompt": payload.get("prompt", ""),
                "style_tags": payload.get("style_tags", []),
                "panels": len(panels),
            },
            "generation": {
                "total_time": total_time,
                "panels_completed": len(
                    [p for p in panels if p.status == "completed"]
                ),
                "panels_failed": len(
                    [p for p in panels if p.status == "failed"]
                ),
                "errors": errors,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def _save_current_state(
        self,
        message: Dict[str, Any],
        panels: List[PanelContent],
        metadata: Dict[str, Any],
    ) -> None:
        """Save current state to session storage"""
        if not self.session_manager:
            return

        state_data = {
            "current_message": message,
            "panels": [panel.to_dict() for panel in panels],
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        state_file = Path(
            f"storyboard/{self.current_session}/agents/bayko_state.json"
        )
        state_file.parent.mkdir(parents=True, exist_ok=True)

        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get current generation statistics"""
        return self.generation_stats.copy()

    def _initialize_session(
        self, session_id: str, conversation_id: Optional[str] = None
    ):
        """Initialize session with unified memory system (matches Brown's pattern)"""
        self.current_session = session_id
        conversation_id = conversation_id or f"conv_{session_id}"

        # Initialize session-specific services (matches Brown's pattern)
        self.session_manager = ServiceSessionManager(
            session_id, conversation_id
        )
        self.message_factory = MessageFactory(session_id, conversation_id)
        self.memory = AgentMemory(session_id, "bayko")

        # Create Bayko's directory structure
        self._create_session_directories(session_id)

        print(f"🧠 Bayko initialized unified memory for session {session_id}")
        logger.info(f"Bayko session initialized: {session_id}")

    def _create_session_directories(self, session_id: str):
        """Create Bayko's session directory structure"""
        session_dir = Path(f"storyboard/{session_id}")
        content_dir = session_dir / "content"
        agents_dir = session_dir / "agents"
        iterations_dir = session_dir / "iterations"
        output_dir = session_dir / "output"

        # Create directory structure
        for dir_path in [content_dir, agents_dir, iterations_dir, output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def update_metadata(self, metadata: Dict[str, Any]):
        """Update content metadata"""
        if not self.current_session:
            return

        session_dir = Path(f"storyboard/{self.current_session}")
        content_dir = session_dir / "content"
        content_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = content_dir / "metadata.json"

        # Load existing metadata if it exists
        existing_metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                existing_metadata = json.load(f)

        # Merge with new metadata
        existing_metadata.update(metadata)
        existing_metadata["updated_at"] = datetime.utcnow().isoformat() + "Z"

        with open(metadata_file, "w") as f:
            json.dump(existing_metadata, f, indent=2)

    def save_iteration_data(self, iteration: int, data: Dict[str, Any]):
        """Save iteration-specific data"""
        if not self.current_session:
            return

        session_dir = Path(f"storyboard/{self.current_session}")
        iterations_dir = session_dir / "iterations"
        iterations_dir.mkdir(parents=True, exist_ok=True)
        iteration_file = iterations_dir / f"v{iteration}_generation.json"

        with open(iteration_file, "w") as f:
            json.dump(data, f, indent=2)

    def save_bayko_state(self, state_data: Dict[str, Any]):
        """Save Bayko's current state"""
        if not self.current_session:
            return

        session_dir = Path(f"storyboard/{self.current_session}")
        agents_dir = session_dir / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        state_file = agents_dir / "bayko_state.json"

        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information (matches Brown's interface)"""
        memory_size = 0
        if self.memory:
            try:
                memory_size = self.memory.get_memory_size()
            except:
                memory_size = 0

        return {
            "session_id": self.current_session,
            "memory_size": memory_size,
            "generation_stats": self.generation_stats,
        }

    def _improve_prompt_with_feedback(
        self,
        original_prompt: str,
        feedback: Dict[str, Any],
        improvements: List[str],
    ) -> str:
        """
        Use mini LLM to improve a prompt based on feedback

        Args:
            original_prompt: The original prompt to improve
            feedback: Dictionary containing feedback data
            improvements: List of specific improvements to make

        Returns:
            Improved prompt
        """
        if not self.llm:
            # Fallback to original prompt if LLM is not initialized
            logger.warning("LLM not available, using original prompt")
            return original_prompt

        try:
            # Construct feedback context
            feedback_str = ""
            if feedback:
                feedback_str = "Feedback:\n"
                for key, value in feedback.items():
                    feedback_str += f"- {key}: {value}\n"

            improvements_str = ""
            if improvements:
                improvements_str = "Requested improvements:\n"
                for imp in improvements:
                    improvements_str += f"- {imp}\n"

            system_prompt = """You are an expert at refining SDXL image generation prompts.
            Analyze the feedback and improve the original prompt while maintaining its core narrative elements.
            Focus on addressing the specific feedback points and requested improvements.
            Return ONLY the improved prompt with no additional formatting or explanations."""

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Original prompt: {original_prompt}\n\n{feedback_str}\n{improvements_str}",
                    },
                ],
                temperature=0.5,
                max_tokens=500,
            )

            if response.choices[0].message.content:
                return response.choices[0].message.content.strip()

            # Fallback if no valid response
            return original_prompt

        except Exception as e:
            logger.error(f"Failed to improve prompt: {e}")
            # Fallback to original prompt
            return original_prompt


# Example usage and testing
async def main():
    """Example usage of Agent Bayko leveraging the BaykoWorkflow"""
    from agents.bayko_workflow import create_agent_bayko
    import os

    # Create Bayko agent using the factory function
    bayko_workflow = create_agent_bayko(os.getenv("OPENAI_API_KEY"))

    # Initialize a test session
    session_id = "test_session_001"
    conversation_id = "conv_001"
    bayko_workflow.initialize_session(session_id, conversation_id)

    test_message = {
        "message_id": "msg_12345",
        "timestamp": "2025-01-15T10:30:00Z",
        "sender": "agent_brown",
        "recipient": "agent_bayko",
        "message_type": "generation_request",
        "payload": {
            "prompt": "A moody K-pop idol finds a puppy on the street. It changes everything. Visual style: whimsical, nature, soft_lighting, watercolor, mood: peaceful, color palette: warm_earth_tones",
            "original_prompt": "A moody K-pop idol finds a puppy on the street. It changes everything.",
            "style_tags": [
                "whimsical",
                "nature",
                "soft_lighting",
                "watercolor",
            ],
            "panels": 4,
            "language": "korean",
            "extras": ["narration", "subtitles"],
            "style_config": {
                "primary_style": "studio_ghibli",
                "mood": "peaceful",
                "color_palette": "warm_earth_tones",
                "confidence": 0.9,
            },
            "generation_params": {
                "quality": "high",
                "aspect_ratio": "16:9",
                "panel_layout": "sequential",
            },
        },
        "context": {
            "conversation_id": conversation_id,
            "session_id": session_id,
            "iteration": 1,
            "previous_feedback": None,
            "validation_score": 0.85,
        },
    }

    # Process generation request
    print("🎨 Processing generation request...")
    result = bayko_workflow.process_generation_request(test_message)

    # Print generation results
    print("\n✨ Generation completed!")
    print("-" * 40)
    print(f"Result type: {type(result)}")

    if isinstance(result, str):
        # For workflow result
        print(f"Generation Result: {result}")
    else:
        # For direct Bayko result
        result_data = result.payload
        print(f"\nPanels Generated: {len(result_data['panels'])}")
        print(
            f"Total Time: {result_data['metadata']['generation']['total_time']:.2f}s"
        )
        print(
            f"Success Rate: {result_data['metadata']['generation']['panels_completed']}/{len(result_data['panels'])}"
        )

        if isinstance(result_data, dict) and result.get("metadata", {}).get(
            "generation", {}
        ).get("errors"):

            print("\n⚠️ Errors:")
            for error in result_data["metadata"]["generation"]["errors"]:
                print(f"- {error}")

    print("\n✅ Test operations completed successfully!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
