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
import os

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are already set

# OpenAI for LLM functionality
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Core services - Updated to match Brown's memory system
from services.unified_memory import AgentMemory
from services.session_manager import SessionManager as ServiceSessionManager
from services.message_factory import MessageFactory, AgentMessage, MessageType
from agents.bayko_tools import (
    ModalImageGenerator,
    TTSGenerator,
    SubtitleGenerator,
)

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

    def __init__(self, llm: Optional[OpenAI] = None):
        # Initialize content generators
        self.image_generator = ModalImageGenerator()
        self.tts_generator = TTSGenerator()
        self.subtitle_generator = SubtitleGenerator()

        # LLM for prompt generation and refinement
        self.llm = llm
        if self.llm:
            logger.info("Agent Bayko initialized with LLM support")
        else:
            logger.info(
                "Agent Bayko initialized without LLM - using fallback methods"
            )

        # Processing state
        self.current_session = None
        self.session_manager = None
        self.generation_stats = {
            "total_panels_generated": 0,
            "total_generation_time": 0.0,
            "average_panel_time": 0.0,
            "error_count": 0,
        }

        # Memory will be initialized per session (matches Brown's pattern)
        self.memory = None
        self.session_manager = None
        self.message_factory = None

        logger.info("Agent Bayko initialized with unified memory support")

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

        # Initialize session (matches Brown's pattern)
        self._initialize_session(session_id, context.get("conversation_id"))

        # Log received request to memory
        self.memory.add_message(
            "user", f"Received generation request: {payload.get('prompt', '')}"
        )

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
        self.update_metadata(metadata)

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

        # Create GenerationResult for internal use
        generation_result = GenerationResult(
            session_id=session_id,
            panels=panels,
            metadata=metadata,
            status=status,
            total_time=total_time,
            errors=errors,
        )

        # Create AgentMessage response using MessageFactory
        response_message = self.message_factory.create_approval_message(
            generation_result.to_dict(),
            {
                "overall_score": 1.0 if not errors else 0.7,
                "generation_successful": True,
                "panels_generated": len(panels),
                "total_time": total_time,
            },
            1,  # iteration
        )

        # Save conversation log
        if self.session_manager:
            self.session_manager._save_conversation_log(response_message)

        # Log completion to memory
        if self.memory:
            self.memory.add_message(
                "assistant",
                f"Generated {len(panels)} panels successfully in {total_time:.2f}s",
            )

        logger.info(
            f"Generation completed for session {session_id} in {total_time:.2f}s"
        )
        return response_message

    async def process_refinement_request(
        self, message: Dict[str, Any]
    ) -> AgentMessage:
        """
        Process refinement request from Agent Brown

        Args:
            message: AgentMessage from Brown containing refinement request

        Returns:
            AgentMessage with refined content
        """
        start_time = time.time()

        payload = message.get("payload", {})
        context = message.get("context", {})
        session_id = context.get("session_id")

        logger.info(f"Processing refinement request for session {session_id}")

        # Initialize session if not already done
        if self.current_session != session_id:
            self._initialize_session(
                session_id, context.get("conversation_id")
            )

        # Extract refinement data
        original_content = payload.get("original_content", {})
        feedback = payload.get("feedback", {})
        improvements = payload.get("specific_improvements", [])
        focus_areas = payload.get("focus_areas", [])
        iteration = payload.get("iteration", 1)

        # Log refinement request to memory
        self.memory.add_message(
            "user",
            f"Received refinement request for iteration {iteration}: {', '.join(improvements)}",
        )

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
        self.update_metadata(metadata)

        # Save iteration data
        self.save_iteration_data(
            iteration,
            {
                "refinement_request": payload,
                "applied_improvements": improvements,
                "focus_areas": focus_areas,
                "refined_panels": [asdict(panel) for panel in refined_panels],
                "processing_time": total_time,
            },
        )

        # Create GenerationResult for internal use
        generation_result = GenerationResult(
            session_id=session_id,
            panels=refined_panels,
            metadata=metadata,
            status=GenerationStatus.COMPLETED,
            total_time=total_time,
            errors=[],
            refinement_applied=True,
        )

        # Create AgentMessage response using MessageFactory
        response_message = self.message_factory.create_approval_message(
            generation_result.to_dict(),
            {
                "overall_score": 1.0,
                "refinement_successful": True,
                "improvements_applied": improvements,
                "total_time": total_time,
            },
            iteration,
        )

        # Save conversation log
        if self.session_manager:
            self.session_manager._save_conversation_log(response_message)

        # Log refinement completion to memory
        if self.memory:
            self.memory.add_message(
                "assistant",
                f"Completed refinement with {len(improvements)} improvements in {total_time:.2f}s",
            )

        logger.info(
            f"Refinement completed for session {session_id} in {total_time:.2f}s"
        )
        return response_message

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

    def generate_prompt_from_description(
        self, description: str, style_tags: List[str], mood: str
    ) -> str:
        """
        Generate a detailed prompt using LLM based on description, style tags, and mood
        Following Agent Brown's pattern for session management and logging

        Args:
            description: Panel description
            style_tags: Visual style tags
            mood: Mood for the panel

        Returns:
            Generated prompt string
        """
        if not self.llm:
            logger.warning(
                "No LLM available for prompt generation, using fallback"
            )
            if self.memory:
                self.memory.add_message(
                    "assistant",
                    "LLM not available - using fallback prompt generation",
                )
            return description

        try:
            # Get session context for better prompts (like Brown does)
            session_context = ""
            if self.memory:
                history = self.memory.get_history()
                if history:
                    recent_context = [msg["content"] for msg in history[-5:]]
                    session_context = " | ".join(recent_context)

            # Construct detailed LLM prompt with structured output requirements
            llm_prompt = f"""You are an expert comic panel designer working with Agent Brown in a multi-agent comic generation system. Your task is to create a vivid, detailed prompt suitable for SDXL text-to-image generation.

CONTEXT FROM AGENT BROWN:
- Session Context: {session_context}
- Panel Description: "{description}"
- Style Tags: {style_tags}
- Mood: "{mood}"

REQUIREMENTS:
1. Create a detailed visual prompt that incorporates ALL style tags: {style_tags}
2. Ensure the mood "{mood}" is clearly conveyed through visual elements
3. Include specific details about lighting, composition, and atmosphere
4. Keep the prompt under 150 words but rich in visual detail
5. Ensure compatibility with SDXL image generation
6. Account for all metadata provided by Agent Brown

OUTPUT FORMAT:
Return only the enhanced prompt text, no explanations or additional text.

EXAMPLE OUTPUT:
"A melancholic K-pop idol in designer streetwear walking alone through rain-soaked Seoul streets at twilight, soft golden streetlight creating dramatic shadows, whimsical watercolor style with soft lighting effects, peaceful mood conveyed through gentle rain droplets and warm earth tones, cinematic composition with shallow depth of field"

Generate the enhanced prompt now:"""

            # Call the LLM with detailed system prompt
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert comic panel designer who creates vivid, detailed prompts for SDXL text-to-image models. You work closely with Agent Brown to ensure all metadata and style requirements are incorporated. Always output only the enhanced prompt text without explanations.",
                    },
                    {"role": "user", "content": llm_prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )

            generated_prompt = response.choices[0].message.content.strip()
            logger.info(
                f"Successfully generated LLM prompt for panel: {generated_prompt[:50]}..."
            )

            # Log detailed information to memory and session (following Brown's pattern)
            if self.memory:
                self.memory.add_message(
                    "assistant", f"LLM enhanced prompt: {generated_prompt}"
                )
                self.memory.add_message(
                    "system", f"Style tags applied: {style_tags}, Mood: {mood}"
                )

            # Save LLM generation data to session (following Brown's session management)
            if hasattr(self, "current_session") and self.current_session:
                llm_data = {
                    "original_description": description,
                    "style_tags": style_tags,
                    "mood": mood,
                    "generated_prompt": generated_prompt,
                    "llm_model": "gpt-4o-mini",
                    "generation_timestamp": datetime.utcnow().isoformat()
                    + "Z",
                }
                self._save_llm_generation_data(llm_data)

            return generated_prompt

        except Exception as e:
            error_msg = f"LLM prompt generation failed: {str(e)}"
            logger.error(error_msg)
            if self.memory:
                self.memory.add_message(
                    "assistant",
                    f"LLM generation failed, using fallback: {str(e)}",
                )
            return description

    def revise_panel_description(
        self, description: str, feedback: Dict, focus_areas: List[str]
    ) -> str:
        """
        Revise panel description using LLM based on feedback
        Following Agent Brown's pattern for detailed feedback processing

        Args:
            description: Original panel description
            feedback: Feedback dictionary from Brown
            focus_areas: Areas that need focus

        Returns:
            Revised description string
        """
        if not self.llm:
            logger.warning(
                "No LLM available for description revision, using fallback"
            )
            if self.memory:
                self.memory.add_message(
                    "assistant", "LLM not available - using fallback revision"
                )
            return self._apply_description_improvements(
                description,
                feedback.get("improvement_suggestions", []),
                focus_areas,
            )

        try:
            # Get comprehensive message history for context (like Brown does)
            message_history = ""
            brown_feedback_context = ""
            if self.memory:
                history = self.memory.get_history()
                recent_messages = history[-10:]  # Last 10 messages for context
                message_history = "\n".join(
                    [
                        f"{msg['role']}: {msg['content']}"
                        for msg in recent_messages
                    ]
                )

                # Extract Brown's feedback patterns
                brown_messages = [
                    msg
                    for msg in recent_messages
                    if "brown" in msg.get("content", "").lower()
                ]
                if brown_messages:
                    brown_feedback_context = "\n".join(
                        [msg["content"] for msg in brown_messages[-3:]]
                    )

            # Construct detailed LLM prompt for revision with structured requirements
            llm_prompt = f"""You are an expert comic panel designer working with Agent Brown to refine comic content. Agent Brown has provided feedback that requires improvement. Your task is to generate an improved panel description that addresses all feedback while maintaining story coherence.

ORIGINAL PANEL DESCRIPTION:
"{description}"

AGENT BROWN'S FEEDBACK:
{feedback}

FOCUS AREAS REQUIRING IMPROVEMENT:
{focus_areas}

RECENT CONVERSATION HISTORY:
{message_history}

BROWN'S SPECIFIC FEEDBACK CONTEXT:
{brown_feedback_context}

REQUIREMENTS FOR REVISION:
1. Address each focus area: {focus_areas}
2. Incorporate feedback suggestions: {feedback.get('improvement_suggestions', [])}
3. Maintain the core story elements and character consistency
4. Enhance visual details that support the requested improvements
5. Ensure the revision aligns with the overall session context
6. Keep the description concise but vivid (under 100 words)

OUTPUT FORMAT:
Return only the improved panel description, no explanations or additional text.

EXAMPLE OUTPUT:
"A melancholic K-pop idol in designer streetwear walking alone through rain-soaked Seoul streets at twilight, enhanced visual style consistency through unified color palette, improved narrative flow with subtle body language showing emotional transformation, soft golden streetlight creating dramatic shadows"

Generate the improved panel description now:"""

            # Call the LLM with detailed system prompt for revision
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert comic panel designer who refines descriptions based on Agent Brown's feedback to improve visual storytelling. You understand the multi-agent workflow and ensure all improvements align with the session context. Always output only the improved description without explanations.",
                    },
                    {"role": "user", "content": llm_prompt},
                ],
                max_tokens=150,
                temperature=0.6,
            )

            revised_description = response.choices[0].message.content.strip()
            logger.info(
                f"Successfully revised description using LLM: {revised_description[:50]}..."
            )

            # Log detailed revision information to memory and session (following Brown's pattern)
            if self.memory:
                self.memory.add_message(
                    "assistant",
                    f"LLM revised description: {revised_description}",
                )
                self.memory.add_message(
                    "system", f"Addressed focus areas: {focus_areas}"
                )
                self.memory.add_message(
                    "system",
                    f"Applied improvements: {feedback.get('improvement_suggestions', [])}",
                )

            # Save revision data to session
            if hasattr(self, "current_session") and self.current_session:
                revision_data = {
                    "original_description": description,
                    "feedback": feedback,
                    "focus_areas": focus_areas,
                    "revised_description": revised_description,
                    "llm_model": "gpt-4o-mini",
                    "revision_timestamp": datetime.utcnow().isoformat() + "Z",
                }
                self._save_llm_revision_data(revision_data)

            return revised_description

        except Exception as e:
            error_msg = f"LLM description revision failed: {str(e)}"
            logger.error(error_msg)
            if self.memory:
                self.memory.add_message(
                    "assistant",
                    f"LLM revision failed, using fallback: {str(e)}",
                )
            return self._apply_description_improvements(
                description,
                feedback.get("improvement_suggestions", []),
                focus_areas,
            )

    def _save_llm_generation_data(self, llm_data: Dict[str, Any]):
        """Save LLM generation data to session (following Brown's session management pattern)"""
        try:
            session_dir = Path(f"storyboard/{self.current_session}")
            llm_dir = session_dir / "llm_data"
            llm_dir.mkdir(parents=True, exist_ok=True)

            # Save generation data with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            generation_file = llm_dir / f"generation_{timestamp}.json"

            with open(generation_file, "w") as f:
                json.dump(llm_data, f, indent=2)

            logger.info(f"Saved LLM generation data to {generation_file}")
        except Exception as e:
            logger.error(f"Failed to save LLM generation data: {e}")

    def _save_llm_revision_data(self, revision_data: Dict[str, Any]):
        """Save LLM revision data to session (following Brown's session management pattern)"""
        try:
            session_dir = Path(f"storyboard/{self.current_session}")
            llm_dir = session_dir / "llm_data"
            llm_dir.mkdir(parents=True, exist_ok=True)

            # Save revision data with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            revision_file = llm_dir / f"revision_{timestamp}.json"

            with open(revision_file, "w") as f:
                json.dump(revision_data, f, indent=2)

            logger.info(f"Saved LLM revision data to {revision_file}")
        except Exception as e:
            logger.error(f"Failed to save LLM revision data: {e}")

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
            # Show progress and log to memory
            print(f"🎨 Bayko generating panel {panel_id}...")
            if self.memory:
                self.memory.add_message(
                    "assistant",
                    f"Starting generation for panel {panel_id}: {description}",
                )

            # Generate enhanced prompt using LLM if available (following Brown's pattern)
            mood = "neutral"

            # Try to extract mood from enhanced_prompt or use default
            if "mood:" in enhanced_prompt.lower():
                mood_part = (
                    enhanced_prompt.lower()
                    .split("mood:")[1]
                    .split(",")[0]
                    .strip()
                )
                mood = mood_part if mood_part else "neutral"

            # Generate LLM-enhanced prompt or use fallback
            if self.llm:
                logger.info(
                    f"Using LLM to generate enhanced prompt for panel {panel_id}"
                )
                if self.memory:
                    self.memory.add_message(
                        "system",
                        f"Generating LLM-enhanced prompt for panel {panel_id}",
                    )

                llm_enhanced_prompt = self.generate_prompt_from_description(
                    description, style_tags, mood
                )
                final_prompt = f"{enhanced_prompt}. {llm_enhanced_prompt}"

                if self.memory:
                    self.memory.add_message(
                        "assistant",
                        f"Panel {panel_id} using LLM-enhanced prompt",
                    )
            else:
                logger.info(
                    f"Using fallback prompt generation for panel {panel_id}"
                )
                if self.memory:
                    self.memory.add_message(
                        "system",
                        f"Panel {panel_id} using fallback prompt (no LLM)",
                    )
                final_prompt = f"{enhanced_prompt}. {description}"

            # Generate image
            print(f"  🖼️  Using SDXL tool for panel {panel_id}")
            logger.info(f"Generating image for panel {panel_id}")
            image_path, image_time = (
                await self.image_generator.generate_panel_image(
                    final_prompt,
                    style_tags,
                    panel_id,
                    session_id,
                )
            )
            panel.image_path = image_path
            panel.style_applied = style_tags.copy()

            # Generate audio if requested
            if "narration" in extras:
                print(f"  🎵 Using TTS tool for panel {panel_id}")
                logger.info(f"Generating audio for panel {panel_id}")
                audio_path, audio_time = (
                    await self.tts_generator.generate_narration(
                        description, language, panel_id, session_id
                    )
                )
                panel.audio_path = audio_path

                # Generate subtitles if requested
                if "subtitles" in extras:
                    print(f"  📝 Generating subtitles for panel {panel_id}")
                    logger.info(f"Generating subtitles for panel {panel_id}")
                    subs_path, subs_time = (
                        await self.subtitle_generator.generate_subtitles(
                            description, audio_time, panel_id, session_id
                        )
                    )
                    panel.subtitles_path = subs_path

            panel.generation_time = time.time() - generation_start
            print(
                f"  ✅ Panel {panel_id} completed in {panel.generation_time:.2f}s"
            )

            # Log successful panel completion to memory
            if self.memory:
                self.memory.add_message(
                    "assistant",
                    f"Panel {panel_id} generated successfully in {panel.generation_time:.2f}s",
                )

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

                # Apply specific improvements to the description using LLM or fallback (following Brown's pattern)
                if self.llm:
                    logger.info(
                        f"Using LLM to revise panel {panel_id} description"
                    )
                    if self.memory:
                        self.memory.add_message(
                            "system",
                            f"Applying LLM-based refinement to panel {panel_id}",
                        )

                    refined_description = self.revise_panel_description(
                        description, feedback, focus_areas
                    )

                    if self.memory:
                        self.memory.add_message(
                            "assistant", f"Panel {panel_id} refined using LLM"
                        )
                else:
                    logger.info(
                        f"Using fallback method to revise panel {panel_id} description"
                    )
                    if self.memory:
                        self.memory.add_message(
                            "system",
                            f"Panel {panel_id} using fallback refinement (no LLM)",
                        )

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
            "memory_history": self.memory.get_history() if self.memory else [],
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }

        # Save using Bayko's state saving method
        self.save_bayko_state(state_data)

        # Log state save to memory
        if self.memory:
            self.memory.add_message(
                "assistant", f"Saved session state with {len(panels)} panels"
            )

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
