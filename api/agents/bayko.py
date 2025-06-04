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

Bayko operates as the backend creative engine, transforming Brown's validated
and structured story plans into actual comic content through AI-powered
image generation, text-to-speech, and subtitle creation.
"""

import json

# import uuid  # Currently unused
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import time

# TODO: Replace with actual Modal imports when available
# import modal

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
    """Content for a single comic panel"""

    panel_id: int
    description: str
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    subtitles_path: Optional[str] = None
    generation_time: Optional[float] = None
    style_applied: Optional[List[str]] = None
    errors: Optional[List[str]] = None

    def __post_init__(self):
        if self.style_applied is None:
            self.style_applied = []
        if self.errors is None:
            self.errors = []


@dataclass
class GenerationResult:
    """Result of content generation process"""

    session_id: str
    panels: List[PanelContent]
    metadata: Dict[str, Any]
    status: GenerationStatus
    total_time: float
    errors: List[str]
    refinement_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "panels": [asdict(panel) for panel in self.panels],
            "metadata": self.metadata,
            "status": self.status.value,
            "total_time": self.total_time,
            "errors": self.errors,
            "refinement_applied": self.refinement_applied,
        }


class ModalImageGenerator:
    """
    Stub implementation for Modal SDXL image generation
    TODO: Replace with actual Modal integration
    """

    def __init__(self):
        self.model_loaded = False
        logger.info("ModalImageGenerator initialized (stub)")

    async def generate_panel_image(
        self,
        prompt: str,
        style_tags: List[str],
        panel_id: int,
        session_id: str,
    ) -> Tuple[str, float]:
        """
        Generate comic panel image using SDXL via Modal

        Args:
            prompt: Enhanced prompt for the panel
            style_tags: Visual style tags to apply
            panel_id: Panel number for naming
            session_id: Session identifier

        Returns:
            Tuple of (image_path, generation_time)
        """
        start_time = time.time()

        # TODO: Replace with actual Modal remote call
        # result = modal_app.generate_comic_panel.remote(prompt, style_tags)

        # Simulate generation time
        await asyncio.sleep(2.0)  # Simulate SDXL generation time

        # Create output path
        content_dir = Path(f"storyboard/{session_id}/content")
        content_dir.mkdir(parents=True, exist_ok=True)
        image_path = content_dir / f"panel_{panel_id}.png"

        # Stub: Create placeholder file
        with open(image_path, "w") as f:
            f.write(f"# Placeholder image for panel {panel_id}\n")
            f.write(f"# Prompt: {prompt}\n")
            f.write(f"# Style: {', '.join(style_tags)}\n")

        generation_time = time.time() - start_time
        logger.info(
            f"Generated image for panel {panel_id} in {generation_time:.2f}s"
        )

        return str(image_path), generation_time


class TTSGenerator:
    """
    Stub implementation for Text-to-Speech generation
    TODO: Replace with actual TTS service integration
    """

    def __init__(self):
        self.voice_models = {
            "english": "en-US-neural",
            "korean": "ko-KR-neural",
            "japanese": "ja-JP-neural",
            "spanish": "es-ES-neural",
            "french": "fr-FR-neural",
        }
        logger.info("TTSGenerator initialized (stub)")

    async def generate_narration(
        self, text: str, language: str, panel_id: int, session_id: str
    ) -> Tuple[str, float]:
        """
        Generate audio narration for panel

        Args:
            text: Text to convert to speech
            language: Target language
            panel_id: Panel number for naming
            session_id: Session identifier

        Returns:
            Tuple of (audio_path, generation_time)
        """
        start_time = time.time()

        # TODO: Replace with actual TTS service call
        # result = tts_service.generate_audio(text, self.voice_models.get(language))

        # Simulate generation time
        await asyncio.sleep(1.0)

        # Create output path
        content_dir = Path(f"storyboard/{session_id}/content")
        content_dir.mkdir(parents=True, exist_ok=True)
        audio_path = content_dir / f"panel_{panel_id}_audio.mp3"

        # Stub: Create placeholder file
        with open(audio_path, "w") as f:
            f.write(f"# Placeholder audio for panel {panel_id}\n")
            f.write(f"# Text: {text}\n")
            f.write(f"# Language: {language}\n")

        generation_time = time.time() - start_time
        logger.info(
            f"Generated audio for panel {panel_id} in {generation_time:.2f}s"
        )

        return str(audio_path), generation_time


class SubtitleGenerator:
    """
    Subtitle generation in VTT format
    """

    def __init__(self):
        logger.info("SubtitleGenerator initialized")

    async def generate_subtitles(
        self, text: str, audio_duration: float, panel_id: int, session_id: str
    ) -> Tuple[str, float]:
        """
        Generate VTT subtitle file for panel

        Args:
            text: Text to create subtitles for
            audio_duration: Duration of audio in seconds
            panel_id: Panel number for naming
            session_id: Session identifier

        Returns:
            Tuple of (subtitles_path, generation_time)
        """
        start_time = time.time()

        # Create output path
        content_dir = Path(f"storyboard/{session_id}/content")
        content_dir.mkdir(parents=True, exist_ok=True)
        subs_path = content_dir / f"panel_{panel_id}_subs.vtt"

        # Generate VTT content
        vtt_content = self._create_vtt_content(text, audio_duration)

        with open(subs_path, "w", encoding="utf-8") as f:
            f.write(vtt_content)

        generation_time = time.time() - start_time
        logger.info(
            f"Generated subtitles for panel {panel_id} in {generation_time:.2f}s"
        )

        return str(subs_path), generation_time

    def _create_vtt_content(self, text: str, duration: float) -> str:
        """Create VTT format subtitle content"""
        # Simple VTT with single subtitle spanning the duration
        return f"""WEBVTT

1
00:00:00.000 --> 00:00:{duration:06.3f}
{text}
"""


class SessionManager:
    """
    Manages session state and file organization following tech_specs.md
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = Path(f"storyboard/{session_id}")
        self.content_dir = self.session_dir / "content"
        self.agents_dir = self.session_dir / "agents"
        self.iterations_dir = self.session_dir / "iterations"
        self.output_dir = self.session_dir / "output"

        # Create directory structure
        for dir_path in [
            self.content_dir,
            self.agents_dir,
            self.iterations_dir,
            self.output_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_bayko_state(self, state_data: Dict[str, Any]):
        """Save Bayko's current state"""
        state_file = self.agents_dir / "bayko_state.json"
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)

    def update_metadata(self, metadata: Dict[str, Any]):
        """Update content metadata"""
        metadata_file = self.content_dir / "metadata.json"

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
        iteration_file = self.iterations_dir / f"v{iteration}_generation.json"
        with open(iteration_file, "w") as f:
            json.dump(data, f, indent=2)


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

    def __init__(self):
        # Initialize content generators
        self.image_generator = ModalImageGenerator()
        self.tts_generator = TTSGenerator()
        self.subtitle_generator = SubtitleGenerator()

        # Processing state
        self.current_session = None
        self.session_manager = None
        self.generation_stats = {
            "total_panels_generated": 0,
            "total_generation_time": 0.0,
            "average_panel_time": 0.0,
            "error_count": 0,
        }

        logger.info("Agent Bayko initialized")

    async def process_generation_request(
        self, message: Dict[str, Any]
    ) -> GenerationResult:
        """
        Process generation request from Agent Brown

        Args:
            message: AgentMessage from Brown containing generation request

        Returns:
            GenerationResult with generated content and metadata
        """
        start_time = time.time()

        # Extract request data
        payload = message.get("payload", {})
        context = message.get("context", {})
        session_id = context.get("session_id")

        if not session_id:
            raise ValueError("No session_id provided in message context")

        self.current_session = session_id
        self.session_manager = SessionManager(session_id)

        logger.info(f"Processing generation request for session {session_id}")

        # Extract generation parameters
        prompt = payload.get("prompt", "")
        original_prompt = payload.get("original_prompt", "")
        style_tags = payload.get("style_tags", [])
        panel_count = payload.get("panels", 4)
        language = payload.get("language", "english")
        extras = payload.get("extras", [])
        style_config = payload.get("style_config", {})

        # Generate panel descriptions
        panel_descriptions = self._create_panel_descriptions(
            original_prompt, panel_count, style_config
        )

        # Generate content for each panel
        panels = []
        errors = []

        for i, description in enumerate(panel_descriptions, 1):
            try:
                panel = await self._generate_panel_content(
                    panel_id=i,
                    description=description,
                    enhanced_prompt=prompt,
                    style_tags=style_tags,
                    language=language,
                    extras=extras,
                    session_id=session_id,
                )
                panels.append(panel)

                # Update progress
                self._update_generation_progress(i, panel_count)

            except Exception as e:
                error_msg = f"Failed to generate panel {i}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                # Create error panel
                error_panel = PanelContent(
                    panel_id=i, description=description, errors=[error_msg]
                )
                panels.append(error_panel)

        # Calculate total time
        total_time = time.time() - start_time

        # Create metadata
        metadata = self._create_generation_metadata(
            payload, panels, total_time, errors
        )

        # Update session metadata
        self.session_manager.update_metadata(metadata)

        # Save Bayko state
        self._save_current_state(message, panels, metadata)

        # Determine status
        status = GenerationStatus.COMPLETED
        if errors:
            status = (
                GenerationStatus.FAILED
                if len(errors) == len(panels)
                else GenerationStatus.COMPLETED
            )

        result = GenerationResult(
            session_id=session_id,
            panels=panels,
            metadata=metadata,
            status=status,
            total_time=total_time,
            errors=errors,
        )

        logger.info(
            f"Generation completed for session {session_id} in {total_time:.2f}s"
        )
        return result

    async def process_refinement_request(
        self, message: Dict[str, Any]
    ) -> GenerationResult:
        """
        Process refinement request from Agent Brown

        Args:
            message: AgentMessage from Brown containing refinement request

        Returns:
            GenerationResult with refined content
        """
        start_time = time.time()

        payload = message.get("payload", {})
        context = message.get("context", {})
        session_id = context.get("session_id")

        logger.info(f"Processing refinement request for session {session_id}")

        # Extract refinement data
        original_content = payload.get("original_content", {})
        feedback = payload.get("feedback", {})
        improvements = payload.get("specific_improvements", [])
        focus_areas = payload.get("focus_areas", [])
        iteration = payload.get("iteration", 1)

        # Initialize session manager
        self.session_manager = SessionManager(session_id)

        # Apply refinements based on feedback
        refined_panels = await self._apply_refinements(
            original_content, feedback, improvements, focus_areas, session_id
        )

        total_time = time.time() - start_time

        # Create refined metadata
        metadata = self._create_refinement_metadata(
            feedback, improvements, refined_panels, total_time, iteration
        )

        # Update session metadata
        self.session_manager.update_metadata(metadata)

        # Save iteration data
        self.session_manager.save_iteration_data(
            iteration,
            {
                "refinement_request": payload,
                "applied_improvements": improvements,
                "focus_areas": focus_areas,
                "refined_panels": [asdict(panel) for panel in refined_panels],
                "processing_time": total_time,
            },
        )

        result = GenerationResult(
            session_id=session_id,
            panels=refined_panels,
            metadata=metadata,
            status=GenerationStatus.COMPLETED,
            total_time=total_time,
            errors=[],
            refinement_applied=True,
        )

        logger.info(
            f"Refinement completed for session {session_id} in {total_time:.2f}s"
        )
        return result

    def _create_panel_descriptions(
        self, prompt: str, panel_count: int, style_config: Dict[str, Any]
    ) -> List[str]:
        """
        Create individual panel descriptions from the main prompt

        Args:
            prompt: Original story prompt
            panel_count: Number of panels to create
            style_config: Style configuration from Brown

        Returns:
            List of panel descriptions
        """
        # TODO: Use AI service to intelligently break down the story
        # For now, create simple sequential descriptions

        mood = style_config.get("mood", "neutral")

        if panel_count == 4:
            # Standard 4-panel comic structure
            descriptions = [
                f"Opening scene: {prompt} - establishing shot with {mood} mood",
                f"Development: Character interaction or discovery, {mood} atmosphere",
                f"Climax: Key moment or realization, heightened {mood} emotion",
                f"Resolution: Conclusion or aftermath, peaceful {mood} tone",
            ]
        elif panel_count == 3:
            descriptions = [
                f"Setup: {prompt} - introduction with {mood} mood",
                f"Conflict: Main action or discovery, intense {mood} emotion",
                f"Resolution: Conclusion, calm {mood} atmosphere",
            ]
        else:
            # Generic approach for other panel counts
            descriptions = []
            for i in range(panel_count):
                part = f"Part {i+1} of {panel_count}"
                descriptions.append(
                    f"{part}: {prompt} - {mood} mood, panel {i+1}"
                )

        return descriptions[:panel_count]

    async def _generate_panel_content(
        self,
        panel_id: int,
        description: str,
        enhanced_prompt: str,
        style_tags: List[str],
        language: str,
        extras: List[str],
        session_id: str,
    ) -> PanelContent:
        """
        Generate all content for a single panel

        Args:
            panel_id: Panel number
            description: Panel description
            enhanced_prompt: Style-enhanced prompt
            style_tags: Visual style tags
            language: Target language
            extras: Additional content to generate
            session_id: Session identifier

        Returns:
            PanelContent with generated assets
        """
        panel = PanelContent(panel_id=panel_id, description=description)
        generation_start = time.time()

        try:
            # Generate image
            logger.info(f"Generating image for panel {panel_id}")
            image_path, image_time = (
                await self.image_generator.generate_panel_image(
                    f"{enhanced_prompt}. {description}",
                    style_tags,
                    panel_id,
                    session_id,
                )
            )
            panel.image_path = image_path
            panel.style_applied = style_tags.copy()

            # Generate audio if requested
            if "narration" in extras:
                logger.info(f"Generating audio for panel {panel_id}")
                audio_path, audio_time = (
                    await self.tts_generator.generate_narration(
                        description, language, panel_id, session_id
                    )
                )
                panel.audio_path = audio_path

                # Generate subtitles if requested
                if "subtitles" in extras:
                    logger.info(f"Generating subtitles for panel {panel_id}")
                    subs_path, subs_time = (
                        await self.subtitle_generator.generate_subtitles(
                            description, audio_time, panel_id, session_id
                        )
                    )
                    panel.subtitles_path = subs_path

            panel.generation_time = time.time() - generation_start
            logger.info(
                f"Panel {panel_id} generated successfully in {panel.generation_time:.2f}s"
            )

        except Exception as e:
            error_msg = f"Error generating panel {panel_id}: {str(e)}"
            logger.error(error_msg)
            if panel.errors is None:
                panel.errors = []
            panel.errors.append(error_msg)
            panel.generation_time = time.time() - generation_start

        return panel

    async def _apply_refinements(
        self,
        original_content: Dict[str, Any],
        feedback: Dict[str, Any],
        improvements: List[str],
        focus_areas: List[str],
        session_id: str,
    ) -> List[PanelContent]:
        """
        Apply refinements based on Brown's feedback

        Args:
            original_content: Original generated content
            feedback: Feedback from Brown
            improvements: Specific improvement suggestions
            focus_areas: Areas that need focus
            session_id: Session identifier

        Returns:
            List of refined PanelContent
        """
        logger.info(f"Applying refinements: {', '.join(improvements)}")

        original_panels = original_content.get("panels", [])
        refined_panels = []

        for panel_data in original_panels:
            panel_id = panel_data.get("panel_id", panel_data.get("id", 0))
            description = panel_data.get("description", "")

            # Determine if this panel needs refinement
            needs_refinement = self._panel_needs_refinement(
                panel_data, feedback, focus_areas
            )

            if needs_refinement:
                logger.info(f"Refining panel {panel_id}")

                # Apply specific improvements to the description
                refined_description = self._apply_description_improvements(
                    description, improvements, focus_areas
                )

                # Regenerate content with improvements
                # TODO: In production, this would selectively regenerate only what needs fixing
                refined_panel = PanelContent(
                    panel_id=panel_id,
                    description=refined_description,
                    image_path=panel_data.get("image_path"),
                    audio_path=panel_data.get("audio_path"),
                    subtitles_path=panel_data.get("subtitles_path"),
                    style_applied=panel_data.get("style_applied", []),
                    errors=[],
                )

                # Simulate refinement time
                await asyncio.sleep(0.5)

            else:
                # Keep original panel
                refined_panel = PanelContent(
                    panel_id=panel_id,
                    description=description,
                    image_path=panel_data.get("image_path"),
                    audio_path=panel_data.get("audio_path"),
                    subtitles_path=panel_data.get("subtitles_path"),
                    style_applied=panel_data.get("style_applied", []),
                    errors=panel_data.get("errors", []),
                )

            refined_panels.append(refined_panel)

        return refined_panels

    def _panel_needs_refinement(
        self,
        panel_data: Dict[str, Any],
        feedback: Dict[str, Any],
        focus_areas: List[str],
    ) -> bool:
        """Determine if a panel needs refinement based on feedback"""
        # Simple heuristic: refine if focus areas indicate issues
        if "style_consistency" in focus_areas:
            return True
        if "narrative_flow" in focus_areas:
            return True
        if feedback.get("overall_score", 1.0) < 0.7:
            return True
        return False

    def _apply_description_improvements(
        self, description: str, improvements: List[str], focus_areas: List[str]
    ) -> str:
        """Apply improvements to panel description"""
        refined = description

        # Apply specific improvements
        for improvement in improvements:
            if "style" in improvement.lower():
                refined += " (enhanced visual style)"
            elif "flow" in improvement.lower():
                refined += " (improved narrative flow)"
            elif "consistency" in improvement.lower():
                refined += " (consistent with overall theme)"

        return refined

    def _create_generation_metadata(
        self,
        payload: Dict[str, Any],
        panels: List[PanelContent],
        total_time: float,
        errors: List[str],
    ) -> Dict[str, Any]:
        """Create metadata for generation session"""
        return {
            "generation_type": "initial",
            "request_payload": payload,
            "panel_count": len(panels),
            "successful_panels": len([p for p in panels if not p.errors]),
            "failed_panels": len([p for p in panels if p.errors]),
            "total_generation_time": total_time,
            "average_panel_time": total_time / len(panels) if panels else 0,
            "style_tags_applied": payload.get("style_tags", []),
            "language": payload.get("language", "english"),
            "extras_generated": payload.get("extras", []),
            "errors": errors,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    def _create_refinement_metadata(
        self,
        feedback: Dict[str, Any],
        improvements: List[str],
        panels: List[PanelContent],
        total_time: float,
        iteration: int,
    ) -> Dict[str, Any]:
        """Create metadata for refinement session"""
        return {
            "generation_type": "refinement",
            "iteration": iteration,
            "feedback_received": feedback,
            "improvements_applied": improvements,
            "panel_count": len(panels),
            "refinement_time": total_time,
            "refined_at": datetime.utcnow().isoformat() + "Z",
        }

    def _update_generation_progress(
        self, current_panel: int, total_panels: int
    ):
        """Update generation progress statistics"""
        progress = (current_panel / total_panels) * 100
        logger.info(
            f"Generation progress: {progress:.1f}% ({current_panel}/{total_panels})"
        )

    def _save_current_state(
        self,
        message: Dict[str, Any],
        panels: List[PanelContent],
        metadata: Dict[str, Any],
    ):
        """Save Bayko's current state"""
        state_data = {
            "session_id": self.current_session,
            "last_message": message,
            "generated_panels": [asdict(panel) for panel in panels],
            "generation_metadata": metadata,
            "generation_stats": self.generation_stats,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }

        if self.session_manager:
            self.session_manager.save_bayko_state(state_data)

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get current generation statistics"""
        return self.generation_stats.copy()


# Factory function for creating Agent Bayko instances
def create_agent_bayko() -> AgentBayko:
    """
    Create and configure Agent Bayko instance

    Returns:
        Configured AgentBayko instance
    """
    return AgentBayko()


# Example usage and testing
async def main():
    """Example usage of Agent Bayko"""
    # Create agent
    bayko = create_agent_bayko()

    # Example message from Brown
    brown_message = {
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
            "conversation_id": "conv_001",
            "session_id": "session_12345",
            "iteration": 1,
            "previous_feedback": None,
            "validation_score": 0.85,
        },
    }

    # Process generation request
    print("Processing generation request...")
    result = await bayko.process_generation_request(brown_message)

    print("\nGeneration completed!")
    print(f"Session: {result.session_id}")
    print(f"Status: {result.status.value}")
    print(f"Panels generated: {len(result.panels)}")
    print(f"Total time: {result.total_time:.2f}s")
    print(f"Errors: {len(result.errors)}")

    # Show panel details
    for panel in result.panels:
        print(f"\nPanel {panel.panel_id}:")
        print(f"  Description: {panel.description}")
        print(f"  Image: {panel.image_path}")
        print(f"  Audio: {panel.audio_path}")
        print(f"  Subtitles: {panel.subtitles_path}")
        print(f"  Generation time: {panel.generation_time:.2f}s")
        if panel.errors:
            print(f"  Errors: {panel.errors}")

    # Example refinement request
    refinement_message = {
        "message_id": "msg_67890",
        "timestamp": "2025-01-15T10:35:00Z",
        "sender": "agent_brown",
        "recipient": "agent_bayko",
        "message_type": "refinement_request",
        "payload": {
            "original_content": result.to_dict(),
            "feedback": {
                "overall_score": 0.65,
                "style_consistency": 0.6,
                "improvement_suggestions": [
                    "Improve visual style consistency",
                    "Enhance narrative flow",
                ],
            },
            "specific_improvements": [
                "Improve visual style consistency",
                "Enhance narrative flow",
            ],
            "focus_areas": ["style_consistency", "narrative_flow"],
            "iteration": 2,
        },
        "context": {
            "conversation_id": "conv_001",
            "session_id": "session_12345",
            "iteration": 2,
            "refinement_reason": "Quality below threshold",
        },
    }

    # Process refinement request
    print("\nProcessing refinement request...")
    refined_result = await bayko.process_refinement_request(refinement_message)

    print("\nRefinement completed!")
    print(f"Status: {refined_result.status.value}")
    print(f"Refinement applied: {refined_result.refinement_applied}")
    print(f"Refinement time: {refined_result.total_time:.2f}s")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get current generation statistics"""
        return self.generation_stats.copy()


# Factory function for creating Agent Bayko instances
def create_agent_bayko() -> AgentBayko:
    """
    Create and configure Agent Bayko instance

    Returns:
        Configured AgentBayko instance
    """
    return AgentBayko()


# Example usage and testing
async def main():
    """Example usage of Agent Bayko"""
    # Create agent
    bayko = create_agent_bayko()

    # Example message from Brown
    brown_message = {
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
            "conversation_id": "conv_001",
            "session_id": "session_12345",
            "iteration": 1,
            "previous_feedback": None,
            "validation_score": 0.85,
        },
    }

    # Process generation request
    print("Processing generation request...")
    result = await bayko.process_generation_request(brown_message)

    print(f"\nGeneration completed!")
    print(f"Session: {result.session_id}")
    print(f"Status: {result.status.value}")
    print(f"Panels generated: {len(result.panels)}")
    print(f"Total time: {result.total_time:.2f}s")
    print(f"Errors: {len(result.errors)}")

    # Show panel details
    for panel in result.panels:
        print(f"\nPanel {panel.panel_id}:")
        print(f"  Description: {panel.description}")
        print(f"  Image: {panel.image_path}")
        print(f"  Audio: {panel.audio_path}")
        print(f"  Subtitles: {panel.subtitles_path}")
        print(f"  Generation time: {panel.generation_time:.2f}s")
        if panel.errors:
            print(f"  Errors: {panel.errors}")

    # Example refinement request
    refinement_message = {
        "message_id": "msg_67890",
        "timestamp": "2025-01-15T10:35:00Z",
        "sender": "agent_brown",
        "recipient": "agent_bayko",
        "message_type": "refinement_request",
        "payload": {
            "original_content": result.to_dict(),
            "feedback": {
                "overall_score": 0.65,
                "style_consistency": 0.6,
                "improvement_suggestions": [
                    "Improve visual style consistency",
                    "Enhance narrative flow",
                ],
            },
            "specific_improvements": [
                "Improve visual style consistency",
                "Enhance narrative flow",
            ],
            "focus_areas": ["style_consistency", "narrative_flow"],
            "iteration": 2,
        },
        "context": {
            "conversation_id": "conv_001",
            "session_id": "session_12345",
            "iteration": 2,
            "refinement_reason": "Quality below threshold",
        },
    }

    # Process refinement request
    print("\nProcessing refinement request...")
    refined_result = await bayko.process_refinement_request(refinement_message)

    print(f"\nRefinement completed!")
    print(f"Status: {refined_result.status.value}")
    print(f"Refinement applied: {refined_result.refinement_applied}")
    print(f"Refinement time: {refined_result.total_time:.2f}s")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
