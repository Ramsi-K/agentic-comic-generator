"""
Tool wrapper class for Agent Bayko's LLM-enhanced workflow methods
"""

import json
import asyncio
from typing import Dict, Any, List
from llama_index.core.tools import FunctionTool

from agents.bayko import AgentBayko
from agents.bayko_tools import ModalImageGenerator, ModalCodeExecutor


class BaykoWorkflowTools:
    """Tool wrapper class for Agent Bayko's LLM-enhanced methods"""

    def __init__(self, bayko_agent: AgentBayko):
        self.bayko = bayko_agent

    def generate_enhanced_prompt_tool(
        self, description: str, style_tags: str = "[]", mood: str = "neutral"
    ) -> str:
        """Generate LLM-enhanced prompt for SDXL image generation from panel description."""
        try:
            style_tags_list = json.loads(style_tags) if style_tags else []
        except:
            style_tags_list = []

        result = self.bayko.generate_prompt_from_description(
            description, style_tags_list, mood
        )

        return json.dumps(
            {
                "enhanced_prompt": result,
                "original_description": description,
                "style_tags": style_tags_list,
                "mood": mood,
                "llm_used": self.bayko.llm is not None,
            }
        )

    def revise_panel_description_tool(
        self, description: str, feedback: str = "{}", focus_areas: str = "[]"
    ) -> str:
        """Revise panel description based on Agent Brown's feedback using LLM."""
        try:
            feedback_dict = json.loads(feedback) if feedback else {}
            focus_areas_list = json.loads(focus_areas) if focus_areas else []
        except:
            feedback_dict = {}
            focus_areas_list = []

        result = self.bayko.revise_panel_description(
            description, feedback_dict, focus_areas_list
        )

        return json.dumps(
            {
                "revised_description": result,
                "original_description": description,
                "feedback_applied": feedback_dict,
                "focus_areas": focus_areas_list,
                "llm_used": self.bayko.llm is not None,
            }
        )

    async def generate_panel_content_tool(self, panel_data: str) -> str:
        """Generate complete panel content including image, audio, subtitles, and code execution concurrently."""
        try:
            data = json.loads(panel_data)
        except:
            return json.dumps({"error": "Invalid panel data JSON"})

        # Extract panel information
        panel_id = data.get("panel_id", 1)
        description = data.get("description", "")
        enhanced_prompt = data.get("enhanced_prompt", "")
        style_tags = data.get("style_tags", [])
        # language = data.get("language", "english")
        # extras = data.get("extras", [])
        session_id = data.get("session_id", "default")
        # dialogues = data.get("dialogues", [])
        # code_snippets = data.get("code_snippets", [])

        # Initialize Modal tools

        image_gen = ModalImageGenerator()
        # tts_gen = TTSGenerator()
        # subtitle_gen = SubtitleGenerator()
        code_executor = ModalCodeExecutor()

        # Create concurrent tasks for parallel execution
        tasks = []

        # 1. Always generate image
        tasks.append(
            image_gen.generate_panel_image(
                enhanced_prompt, style_tags, panel_id, session_id
            )
        )

        # 4. Execute code if provided
        code_snippets = data.get("code_snippets", [])
        if code_snippets and panel_id <= len(code_snippets):
            code_data = (
                code_snippets[panel_id - 1]
                if isinstance(code_snippets, list)
                else code_snippets
            )
            if isinstance(code_data, dict):
                code = code_data.get("code", "")
                code_language = code_data.get("language", "python")
                context = code_data.get("context", description)
            else:
                code = str(code_data)
                code_language = "python"
                context = description

            if code.strip():
                tasks.append(code_executor.execute_code(context, session_id))
            else:
                tasks.append(
                    asyncio.create_task(asyncio.sleep(0))
                )  # No-op task
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # No-op task

        # Execute all tasks concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = asyncio.get_event_loop().time() - start_time

        # Process results safely
        def safe_get_path(result):
            if isinstance(result, Exception) or result is None:
                return None
            if isinstance(result, tuple) and len(result) >= 1:
                return result[0]
            return None

        def safe_check_exists(result):
            path = safe_get_path(result)
            return path is not None

        image_path = safe_get_path(results[0])
        audio_path = safe_get_path(results[1])
        subtitle_path = safe_get_path(results[2])
        code_path = safe_get_path(results[3])

        # Build result
        result = {
            "panel_id": panel_id,
            "description": description,
            "enhanced_prompt": enhanced_prompt,
            "image_path": image_path,
            "image_url": f"file://{image_path}" if image_path else None,
            "audio_path": audio_path,
            "subtitles_path": subtitle_path,
            "code_result_path": code_path,
            "style_applied": style_tags,
            "generation_time": total_time,
            "status": "completed",
            "concurrent_execution": True,
            "tasks_completed": {
                "image": image_path is not None,
                "audio": audio_path is not None,
                "subtitles": subtitle_path is not None,
                "code": code_path is not None,
            },
        }

        return json.dumps(result)

    def get_session_info_tool(self) -> str:
        """Get current Bayko session information and memory state."""
        info = self.bayko.get_session_info()
        return json.dumps(
            {
                "session_id": info.get("session_id"),
                "memory_size": info.get("memory_size", 0),
                "generation_stats": info.get("generation_stats", {}),
                "llm_available": self.bayko.llm is not None,
                "status": "active" if info.get("session_id") else "inactive",
            }
        )

    def save_llm_data_tool(self, data_type: str, data: str) -> str:
        """Save LLM generation or revision data to session storage."""
        try:
            data_dict = json.loads(data)
        except:
            return json.dumps({"error": "Invalid data JSON"})

        if data_type == "generation":
            self.bayko._save_llm_generation_data(data_dict)
        elif data_type == "revision":
            self.bayko._save_llm_revision_data(data_dict)
        else:
            return json.dumps({"error": "Invalid data type"})

        return json.dumps(
            {
                "status": "saved",
                "data_type": data_type,
                "session_id": self.bayko.current_session,
            }
        )

    def create_llamaindex_tools(self) -> List[FunctionTool]:
        """Create LlamaIndex FunctionTools from Bayko's LLM-enhanced methods"""
        return [
            FunctionTool.from_defaults(
                fn=self.generate_enhanced_prompt_tool,
                name="generate_enhanced_prompt",
                description="Generate LLM-enhanced prompt for SDXL image generation. Takes panel description, style tags, and mood. Returns enhanced prompt optimized for text-to-image models.",
            ),
            FunctionTool.from_defaults(
                fn=self.revise_panel_description_tool,
                name="revise_panel_description",
                description="Revise panel description based on Agent Brown's feedback using LLM. Takes original description, feedback, and focus areas. Returns improved description.",
            ),
            FunctionTool.from_defaults(
                async_fn=self.generate_panel_content_tool,
                name="generate_panel_content",
                description="Generate complete panel content including image, audio, subtitles, and code execution concurrently. Takes panel data JSON with description, style, and generation parameters.",
            ),
            FunctionTool.from_defaults(
                fn=self.get_session_info_tool,
                name="get_session_info",
                description="Get current Bayko session information including memory state and generation statistics.",
            ),
            FunctionTool.from_defaults(
                fn=self.save_llm_data_tool,
                name="save_llm_data",
                description="Save LLM generation or revision data to session storage. Takes data type ('generation' or 'revision') and data JSON.",
            ),
        ]
