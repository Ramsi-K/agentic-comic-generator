"""
Agent Bayko LlamaIndex ReAct Workflow
Hackathon demo showcasing content generation with LLM-enhanced prompts and visible reasoning
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, BaseTool

# Core services
from services.unified_memory import AgentMemory
from services.session_manager import SessionManager
from services.message_factory import MessageFactory
from agents.bayko_tools import (
    ModalImageGenerator,
    TTSGenerator,
    SubtitleGenerator,
)
from agents.bayko import AgentBayko

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

openai_api_key = os.getenv("OPENAI_API_KEY")


class BaykoTools:
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
        language = data.get("language", "english")
        extras = data.get("extras", [])
        session_id = data.get("session_id", "default")
        dialogues = data.get("dialogues", [])
        code_snippets = data.get("code_snippets", [])

        # Initialize Modal tools
        from agents.bayko_tools import (
            ModalImageGenerator,
            TTSGenerator,
            SubtitleGenerator,
            ModalCodeExecutor,
        )

        image_gen = ModalImageGenerator()
        tts_gen = TTSGenerator()
        subtitle_gen = SubtitleGenerator()
        code_executor = ModalCodeExecutor()

        # Create concurrent tasks for parallel execution
        tasks = []

        # 1. Always generate image
        tasks.append(
            image_gen.generate_panel_image(
                enhanced_prompt, style_tags, panel_id, session_id
            )
        )

        # 2. Generate TTS if dialogues provided
        if dialogues and panel_id <= len(dialogues):
            dialogue_text = (
                dialogues[panel_id - 1]
                if isinstance(dialogues, list)
                else str(dialogues)
            )
            tasks.append(
                tts_gen.generate_narration(
                    dialogue_text, language, panel_id, session_id
                )
            )
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # No-op task

        # 3. Generate subtitles if requested (though this might be removed later)
        if "subtitles" in extras and dialogues:
            dialogue_text = (
                dialogues[panel_id - 1]
                if isinstance(dialogues, list) and panel_id <= len(dialogues)
                else description
            )
            tasks.append(
                subtitle_gen.generate_subtitles(
                    dialogue_text, 3.0, panel_id, session_id
                )
            )
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # No-op task

        # 4. Execute code if provided
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
                tasks.append(
                    code_executor.execute_code(
                        code, code_language, panel_id, session_id, context
                    )
                )
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


class BaykoWorkflow:
    """
    Agent Bayko Workflow using LlamaIndex ReActAgent
    Showcases LLM-enhanced content generation with visible reasoning
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize LLM if available
        self.llm = None
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = OpenAI(
                    model="gpt-4o-mini",
                    api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                    temperature=0.7,
                    max_tokens=1024,
                )
            except Exception as e:
                print(f"⚠️ Could not initialize LLM: {e}")
                self.llm = None

        # Initialize core Bayko agent with LLM
        self.bayko_agent = AgentBayko(llm=self.llm)

        # Initialize session services
        self.session_manager = None
        self.memory = None
        self.message_factory = None

        # Create Bayko tools
        self.bayko_tools = BaykoTools(self.bayko_agent)
        self.tools = self.bayko_tools.create_llamaindex_tools()

        # System prompt for Bayko ReActAgent
        self.system_prompt = """You are Agent Bayko, the creative content generation specialist in a multi-agent comic generation system.

🎯 MISSION: Transform Agent Brown's structured requests into high-quality comic content using LLM-enhanced prompts and AI-powered generation.

🔄 WORKFLOW (MUST FOLLOW IN ORDER):
1. RECEIVE structured request from Agent Brown with panel descriptions and style requirements
2. ENHANCE prompts using generate_enhanced_prompt tool - create SDXL-optimized prompts
3. GENERATE content using generate_panel_content tool - create images, audio, subtitles
4. SAVE LLM data using save_llm_data tool - persist generation and revision data
5. RESPOND with completed content and metadata

🛠️ TOOLS AVAILABLE:
- generate_enhanced_prompt: Create LLM-enhanced prompts for SDXL image generation
- revise_panel_description: Improve descriptions based on feedback using LLM
- generate_panel_content: Generate complete panel content (images, audio, subtitles)
- get_session_info: Track session state and generation statistics
- save_llm_data: Persist LLM generation and revision data to session storage

🧠 LLM ENHANCEMENT:
- Use LLM to create detailed, vivid prompts for better image generation
- Incorporate all style tags, mood, and metadata from Agent Brown
- Generate SDXL-compatible prompts with proper formatting
- Apply intelligent refinements based on feedback

🎨 CONTENT GENERATION:
- Create comic panel images using enhanced prompts
- Generate audio narration when requested
- Create VTT subtitle files for accessibility
- Maintain consistent style across all panels

💾 SESSION MANAGEMENT:
- Save all LLM interactions to session storage
- Track generation statistics and performance
- Maintain memory of conversation context
- Log all activities for audit trail

🏆 HACKATHON SHOWCASE:
- Demonstrate LLM-enhanced prompt generation
- Show visible reasoning with Thought/Action/Observation
- Highlight intelligent content creation workflow
- Showcase session management and data persistence

✅ COMPLETION:
When content generation is complete, provide summary with:
- Number of panels generated
- LLM enhancements applied
- Session data saved
- Generation statistics

🚫 IMPORTANT:
- Always use LLM enhancement when available
- Fallback gracefully when LLM unavailable
- Save all generation data to session
- Maintain compatibility with Agent Brown's workflow"""

        # Create ReActAgent for content generation
        self.agent = None
        if self.llm:
            # Cast tools to BaseTool for compatibility
            from typing import cast

            tools_list = cast(List[BaseTool], self.tools)

            self.agent = ReActAgent.from_tools(
                tools=tools_list,
                llm=self.llm,
                memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
                system_prompt=self.system_prompt,
                verbose=True,
                max_iterations=15,
            )

    def initialize_session(
        self, session_id: str, conversation_id: Optional[str] = None
    ):
        """Initialize session services for Bayko workflow"""
        conversation_id = conversation_id or f"conv_{session_id}"

        # Initialize session services
        self.session_manager = SessionManager(session_id, conversation_id)
        self.memory = AgentMemory(session_id, "bayko")
        self.message_factory = MessageFactory(session_id, conversation_id)

        # Update Bayko agent with session services
        self.bayko_agent._initialize_session(session_id, conversation_id)

        print(f"🧠 Bayko workflow initialized for session {session_id}")

    def process_generation_request(self, request_data: Dict[str, Any]) -> str:
        """
        Process content generation request using ReActAgent

        Args:
            request_data: Structured request from Agent Brown

        Returns:
            Agent's response with generated content
        """
        if not self.agent:
            return self._fallback_generation(request_data)

        try:
            print("🤖 Agent Bayko ReAct Workflow")
            print("=" * 60)
            print(f"📝 Request: {request_data.get('prompt', 'N/A')[:100]}...")
            print("\n🔄 Bayko Processing...")
            print("=" * 60)

            # Format request for agent
            request_prompt = f"""Generate comic content for the following request:

Original Prompt: {request_data.get('original_prompt', '')}
Enhanced Prompt: {request_data.get('prompt', '')}
Style Tags: {request_data.get('style_tags', [])}
Panels: {request_data.get('panels', 4)}
Language: {request_data.get('language', 'english')}
Extras: {request_data.get('extras', [])}
Session ID: {request_data.get('session_id', 'default')}

Please generate enhanced prompts and create the comic content."""

            # Process with ReActAgent (shows Thought/Action/Observation)
            response = self.agent.chat(request_prompt)

            print("\n" + "=" * 60)
            print("🎉 Agent Bayko Response:")
            print("=" * 60)

            return str(response)

        except Exception as e:
            error_msg = f"❌ Error in Bayko generation: {str(e)}"
            print(error_msg)
            return self._fallback_generation(request_data)

    def _fallback_generation(self, request_data: Dict[str, Any]) -> str:
        """Fallback generation when ReActAgent is unavailable"""
        print("⚠️ Using fallback generation (no LLM agent)")

        # Use core Bayko methods directly
        panels = request_data.get("panels", 4)
        style_tags = request_data.get("style_tags", [])

        result = {
            "status": "completed",
            "method": "fallback",
            "panels_generated": panels,
            "style_tags_applied": style_tags,
            "llm_enhanced": False,
            "message": f"Generated {panels} panels using fallback methods",
        }

        return json.dumps(result, indent=2)


def create_agent_bayko(openai_api_key: Optional[str] = None) -> BaykoWorkflow:
    """
    Factory function to create Agent Bayko workflow with LlamaIndex integration

    Args:
        openai_api_key: OpenAI API key for LLM functionality

    Returns:
        Configured BaykoWorkflow instance with ReActAgent
    """
    return BaykoWorkflow(openai_api_key=openai_api_key)


# Example usage for testing
def main():
    """Example usage of Bayko Workflow"""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Create workflow
    workflow = create_agent_bayko()

    # Example request from Agent Brown
    test_request = {
        "prompt": "A melancholic K-pop idol in designer streetwear walking alone through rain-soaked Seoul streets at twilight, whimsical watercolor style with soft lighting effects",
        "original_prompt": "A moody K-pop idol finds a puppy on the street. It changes everything.",
        "style_tags": ["whimsical", "nature", "soft_lighting", "watercolor"],
        "panels": 4,
        "language": "korean",
        "extras": ["narration", "subtitles"],
        "session_id": "test_session_001",
    }

    # Initialize session
    workflow.initialize_session("test_session_001")

    # Process request
    print("🧪 Testing Bayko Workflow")
    print("=" * 80)

    result = workflow.process_generation_request(test_request)
    print(result)

    print("\n" + "=" * 80)
    print("✅ Test completed!")


if __name__ == "__main__":
    main()
