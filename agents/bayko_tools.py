import json
import base64

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
import random
import modal
import modal
import random

# Core services - Updated to match Brown's memory system
from services.unified_memory import AgentMemory
from services.session_manager import SessionManager as ServiceSessionManager
from services.message_factory import MessageFactory, AgentMessage, MessageType

# TODO: Replace with actual Modal imports when available
# import modal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModalImageGenerator:
    """
    MCP client implementation for SDXL Turbo image generation via Modal compute
    """

    def __init__(self):
        self.model_loaded = False
        logger.info("ModalImageGenerator initialized")
        # Import the Modal app here to ensure it's only loaded when needed
        try:
            from tools.image_generator import generate_comic_panel, app

            self.generate_panel = generate_comic_panel
            self.app = app
            self.model_loaded = True
        except ImportError as e:
            logger.error(f"Failed to import Modal image generator: {e}")

    async def generate_panel_image(
        self,
        prompt: str,
        style_tags: List[str],  # Kept for backward compatibility but not used
        panel_id: int,
        session_id: str,
    ) -> Tuple[str, float]:
        """
        Generate comic panel image using SDXL via Modal MCP
        Returns tuple of (image_path, generation_time)
        """
        if not self.model_loaded:
            raise RuntimeError(
                "Modal image generator not properly initialized"
            )

        start_time = time.time()
        try:  # Call the Modal function directly
            with self.app.run():
                img_bytes, duration = await self.generate_panel.remote.aio(
                    prompt=prompt,
                    panel_id=panel_id,
                    session_id=session_id,
                    steps=1,  # Using SDXL Turbo default
                    seed=42,
                )

            # Create output path and save the image
            content_dir = Path(f"storyboard/{session_id}/content")
            content_dir.mkdir(parents=True, exist_ok=True)
            image_path = content_dir / f"panel_{panel_id}.png"

            # Save the returned image bytes
            with open(image_path, "wb") as f:
                f.write(img_bytes)

            generation_time = time.time() - start_time
            logger.info(
                f"Generated image for panel {panel_id} in {generation_time:.2f}s"
            )
            return str(image_path), generation_time

        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise


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


class ModalCodeExecutor:
    """
    Modal code execution sandbox using fries.py for Python script generation and execution
    """

    def __init__(self):
        self.app = None
        self.generate_and_run = None
        logger.info("ModalCodeExecutor initializing")
        try:
            from tools.fries import app, generate_and_run_script

            self.app = app
            # Store the Modal function directly like in ModalImageGenerator
            self.generate_and_run = generate_and_run_script
            logger.info("Successfully loaded Modal fries app")
        except ImportError as e:
            logger.error(f"Failed to import Modal fries app: {e}")

    async def execute_code(
        self, prompt: str, session_id: str
    ) -> Tuple[str, float]:
        """
        Execute code in Modal sandbox for interactive comic elements using fries.py

        Args:
            prompt: The prompt to generate and run code for
            session_id: Session identifier for file organization

        Returns:
            Tuple of (script_file_path, execution_time)
        """
        if not self.app or not self.generate_and_run:
            raise RuntimeError("Modal fries app not properly initialized")

        start_time = time.time()

        try:
            # Generate animal art with fries
            animal = random.choice(
                [
                    "cat",
                    "dog",
                    "fish",
                    "bird",
                    "giraffe",
                    "turtle",
                    "monkey",
                    "rabbit",
                    "puppy",
                    "animal",
                ]
            )

            print(
                f"\n🤖 Generating an ascii {animal} for you!"
            )  # Execute code via Modal with the EXACT same prompt structure as main()
            with self.app.run():
                result = await self.generate_and_run.remote.aio(
                    f"""
                    create a simple ASCII art of a {animal}.
                    Create ASCII art using these characters: _ - = ~ ^ \\\\ / ( ) [ ] {{ }} < > | . o O @ *
                    Draw the art line by line with print statements.
                    Write a short, funny Python script.
                    Use only basic Python features.
                    Add a joke or pun about fries in the script.
                    Make it light-hearted and fun.
                    End with a message about fries.
                    Make sure the script runs without errors.
                    """,
                    session_id,
                )

            print("=" * 30)
            print("\n🎮 Code Output:")
            print("=" * 30)
            print("\n\n")
            print(result["output"])

            print("🍟    🍟     🍟")
            print("Golden crispy Python fries")
            print("Coming right up!")
            print()
            print("Haha. Just kidding.")

            script_file = f"storyboard/{session_id}/output/fries_for_you.py"
            os.makedirs(os.path.dirname(script_file), exist_ok=True)

            if result["code"]:
                # Save the generated code locally
                with open(script_file, "w") as f:
                    f.write(result["code"])
                print("\nGo here to check out your actual custom code:")
                print(f"👉 Code saved to: {script_file}")
                print("\n\n\n")

            if result["error"]:
                print("\n❌ Error:")
                print("=" * 40)
                print(result["error"])
                print("Looks like there was an error during execution.")
                print("Here are some extra fries to cheer you up!")
                print("🍟    🍟     🍟")
                print("   🍟     🍟    ")
                print("       🍟      ")
                print("Now with extra machine-learned crispiness.")

            execution_time = time.time() - start_time
            return script_file, execution_time

        except modal.exception.FunctionTimeoutError:
            print(
                "⏰ Script execution timed out after 300 seconds and 3 tries!"
            )
            print("Sorry but codestral is having a hard time drawing today.")
            print("Here's a timeout fry for you! 🍟")
            print("Here are some extra fries to cheer you up!")
            print("🍟    🍟     🍟")
            print("   🍟     🍟    ")
            print("       🍟      ")
            print("Now with extra machine-learned crispiness.")
            raise
        except Exception as e:
            logger.error(f"Failed to execute code: {e}")
            raise
