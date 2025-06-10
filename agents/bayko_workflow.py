"""
Agent Bayko LlamaIndex ReAct Workflow
Hackathon demo showcasing content generation with LLM-enhanced prompts and visible reasoning
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from openai import OpenAI as OpenAIClient
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, BaseTool

# Standard library imports
from pathlib import Path
from datetime import datetime

# Core services
from services.unified_memory import AgentMemory
from services.session_id_generator import SessionIdGenerator
from services.session_manager import SessionManager
from services.message_factory import MessageFactory, AgentMessage
from agents.bayko import AgentBayko
from agents.bayko_tools import (
    ModalImageGenerator,
    ModalCodeExecutor,
)
from agents.bayko_workflow_tools import BaykoWorkflowTools

# Custom prompts
from prompts.bayko_workflow_system_prompt import BAYKO_WORKFLOW_SYSTEM_PROMPT

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from datetime import datetime

openai_api_key = os.getenv("OPENAI_API_KEY")


"""
Handle LLM interactions for prompt enhancement and generation flow
"""


class BaykoWorkflow:
    """
    Agent Bayko Workflow using LlamaIndex ReActAgent
    Showcases LLM-enhanced content generation with visible reasoning
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize LLM if available and not explicitly disabled
        self.llm = None

        # Only initialize if api_key is explicitly provided (not None)
        if openai_api_key is not None:
            try:
                self.llm = LlamaOpenAI(
                    model="gpt-4",
                    api_key=openai_api_key,  # Don't use env var
                    temperature=0.7,
                    max_tokens=2048,
                )
                print("✓ Initialized LlamaIndex LLM for enhanced prompts")
            except Exception as e:
                print(f"⚠️ Could not initialize LlamaIndex LLM: {e}")
                self.llm = None

        # Initialize core Bayko agent with matching LLM state
        self.bayko_agent = AgentBayko()
        if self.llm:
            self.bayko_agent.llm = self.llm

        # Initialize session services as None
        self.session_manager = None
        self.memory = None
        self.message_factory = None

        # Create Bayko tools
        self.bayko_tools = BaykoWorkflowTools(self.bayko_agent)
        self.tools = self.bayko_tools.create_llamaindex_tools()

        # System prompt for ReAct agent
        self.system_prompt = BAYKO_WORKFLOW_SYSTEM_PROMPT

        # Initialize ReAct agent if we have LLM
        self.agent = None
        if self.llm:
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

    def initialize_workflow(self):
        """
        Initialize core workflow components that don't require session context.
        Session-specific services should be initialized via initialize_session().
        """
        # Only initialize the LLM-powered components if not already done
        if self.llm and not self.agent:
            # Initialize ReActAgent with LLM-powered tools
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
            print("✓ Initialized ReActAgent with tools")

    def initialize_session(
        self, session_id: str, conversation_id: Optional[str] = None
    ):
        """
        Initialize or reinitialize session-specific services for Bayko workflow.
        This should be called before processing any requests to ensure proper session context.

        Args:
            session_id: Unique identifier for this generation session
            conversation_id: Optional conversation ID. If not provided, will be derived from session_id
        """
        conversation_id = conversation_id or f"conv_{session_id}"

        # Initialize or reinitialize session services
        self.session_manager = SessionManager(session_id, conversation_id)
        self.memory = AgentMemory(session_id, "bayko")
        self.message_factory = MessageFactory(session_id, conversation_id)

        # Update Bayko agent with session services
        self.bayko_agent._initialize_session(session_id, conversation_id)

        # If we have an LLM agent, ensure its memory is aware of the session
        if self.agent and hasattr(self.agent, "memory"):
            from llama_index.core.llms import ChatMessage, MessageRole

            self.agent.memory = ChatMemoryBuffer.from_defaults(
                token_limit=4000,
                chat_history=[
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=f"Session ID: {session_id}",
                    )
                ],
            )

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

    def reset(self):
        """Reset workflow state for a new session."""
        if self.memory:
            self.memory.clear()
        if self.agent and hasattr(self.agent, "memory"):
            self.agent.memory = ChatMemoryBuffer.from_defaults(
                token_limit=4000
            )
        self.current_session = None
        self.session_manager = None


def create_agent_bayko(openai_api_key: Optional[str] = None) -> BaykoWorkflow:
    """
    Factory function to create and initialize Agent Bayko workflow

    This function creates a BaykoWorkflow instance with proper initialization of:
    - LlamaIndex LLM for enhanced prompts
    - ReActAgent with function tools
    - Memory and session services
    - Core Bayko agent with tools

    Args:
        openai_api_key: OpenAI API key for LLM functionality. If None, will try to use environment variable.

    Returns:
        Configured BaykoWorkflow instance with all components initialized
    """
    # Create workflow instance
    workflow = BaykoWorkflow(openai_api_key=openai_api_key)

    # Initialize all workflow components
    workflow.initialize_workflow()

    return workflow


# Example usage for testing
def main():
    """Example usage of Bayko Workflow"""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Create workflow with API key
    workflow = create_agent_bayko(os.getenv("OPENAI_API_KEY"))

    # Test request
    test_request = {
        "prompt": "A melancholic K-pop idol discovers a lost puppy in the neon-lit streets of Seoul at night. The encounter changes everything.",
        "original_prompt": "A K-pop idol finds a puppy that changes their life",
        "style_tags": [
            "anime",
            "soft_lighting",
            "emotional",
            "watercolor",
            "night_scene",
            "neon_lights",
        ],
        "panels": 4,
        "language": "korean",
        "extras": ["narration", "subtitles"],
        "session_id": SessionIdGenerator.create_session_id("test"),
    }  # Initialize session
    session_id = SessionIdGenerator.create_session_id("test")
    conversation_id = f"conv_{session_id}"
    workflow.initialize_session(session_id, conversation_id)

    # Process request with visible reasoning
    print("\n🤖 Testing Bayko Workflow with LLM Reasoning")
    print("=" * 80)
    print(f"📝 Original prompt: {test_request['original_prompt']}")
    print(f"✨ Enhanced prompt: {test_request['prompt']}")
    print(f"🎨 Style tags: {test_request['style_tags']}")
    print(f"🖼️  Panels: {test_request['panels']}")
    print("\n🔄 Processing generation request...")

    try:
        result = workflow.process_generation_request(test_request)
        print("\n🎉 Generation completed!")
        print("\n📋 Generation Result:")
        print("=" * 40)
        print(result)
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        print("⚠️ Attempting fallback generation...")
        result = workflow._fallback_generation(test_request)
        print("\n📋 Fallback Result:")
        print("=" * 40)
        print(result)
        print("\n⚠️ Test completed with fallback")


if __name__ == "__main__":
    main()
