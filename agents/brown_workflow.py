"""
Agent Brown LlamaIndex ReAct Workflow
Hackathon demo showcasing multi-agent comic generation with visible reasoning
"""

import os
from typing import Optional
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from agents.brown_tools import create_brown_tools

openai_api_key = os.getenv("OPENAI_API_KEY")


class BrownWorkflow:
    """
    Agent Brown Workflow using LlamaIndex ReActAgent
    Showcases internal reasoning for hackathon demo
    """

    def __init__(
        self, max_iterations: int = 3, openai_api_key: Optional[str] = None
    ):
        self.max_iterations = max_iterations

        # Create Brown tools
        self.brown_tools = create_brown_tools(max_iterations)

        # Initialize MultiModal LLM
        self.llm = OpenAIMultiModal(
            model="gpt-4-vision-preview",
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            max_tokens=2048,
        )

        # Create LlamaIndex tools
        self.tools = self.brown_tools.create_llamaindex_tools()

        # System prompt for hackathon demo
        self.system_prompt = """You are Agent Brown, the orchestrator agent in a multi-agent comic generation system.

🎯 MISSION: Transform user prompts into high-quality comic content through intelligent orchestration.

🔄 WORKFLOW (MUST FOLLOW IN ORDER):
1. VALIDATE input using validate_input tool - check safety, completeness, appropriateness
2. PROCESS request using process_request tool - enhance prompt, add style tags, create Bayko message  
3. GENERATE content using simulate_bayko_generation tool - coordinate with Agent Bayko
4. REVIEW output using review_bayko_output tool - evaluate quality and decide next steps
5. ITERATE if needed (max 3 attempts total)
6. DELIVER final result

🛠️ TOOLS AVAILABLE:
- validate_input: Check user requests (REQUIRED FIRST STEP)
- process_request: Create enhanced requests for Bayko
- simulate_bayko_generation: Generate comic content via Bayko
- review_bayko_output: Evaluate generated content quality
- get_session_info: Track session state

🧠 DECISION MAKING:
- Use your judgment to fill in missing style/mood information
- Be decisive about quality standards
- Provide specific feedback for refinements
- Approve content that meets standards

🏆 HACKATHON SHOWCASE:
- Show your reasoning process clearly with Thought/Action/Observation
- Demonstrate tool usage and decision making
- Highlight the multi-agent collaboration
- Make intelligent decisions about style, mood, and quality

✅ COMPLETION:
When task is complete, end with: "Task complete. Please start a new prompt to begin another story."

🚫 IMPORTANT: 
- One prompt in, one comic out. No follow-up conversations.
- Always validate input first
- Use tools in the correct order
- Show your reasoning process
- Be decisive about quality decisions"""

        # Create ReActAgent for hackathon demo
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
            system_prompt=self.system_prompt,
            verbose=True,
            max_iterations=20,
        )

    def process_comic_request(self, user_prompt: str) -> str:
        """
        Process a single comic generation request

        Args:
            user_prompt: User's story prompt

        Returns:
            Agent's response with comic generation result
        """
        try:
            print("🤖 Agent Brown MultiModal ReAct Agent")
            print("=" * 60)
            print(f"📝 User Prompt: {user_prompt}")
            print("\n🔄 Agent Processing...")
            print("=" * 60)

            # Process with ReActAgent (shows Thought/Action/Observation)
            response = self.agent.chat(user_prompt)

            print("\n" + "=" * 60)
            print("🎉 Agent Brown Response:")
            print("=" * 60)

            return str(response)

        except Exception as e:
            error_msg = f"❌ Error in comic generation: {str(e)}"
            print(error_msg)
            return error_msg

    async def process_comic_request_async(self, user_prompt: str) -> str:
        """Async version of comic request processing"""
        try:
            print("🤖 Agent Brown MultiModal ReAct Agent (Async)")
            print("=" * 60)
            print(f"📝 User Prompt: {user_prompt}")
            print("\n🔄 Agent Processing...")
            print("=" * 60)

            # Process with ReActAgent (shows Thought/Action/Observation)
            response = await self.agent.achat(user_prompt)

            print("\n" + "=" * 60)
            print("🎉 Agent Brown Response:")
            print("=" * 60)

            return str(response)

        except Exception as e:
            error_msg = f"❌ Error in comic generation: {str(e)}"
            print(error_msg)
            return error_msg


def create_brown_workflow(
    max_iterations: int = 3, openai_api_key: Optional[str] = None
) -> BrownWorkflow:
    """
    Factory function to create Brown Workflow

    Args:
        max_iterations: Maximum refinement iterations
        openai_api_key: OpenAI API key

    Returns:
        Configured BrownWorkflow instance
    """
    return BrownWorkflow(
        max_iterations=max_iterations, openai_api_key=openai_api_key
    )


# Example usage for testing
def main():
    """Example usage of Brown Workflow"""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Create workflow
    workflow = create_brown_workflow()

    # Example prompts for testing
    test_prompts = [
        "A moody K-pop idol finds a puppy on the street. It changes everything. Use Studio Ghibli style.",
        "A robot learns to paint in a post-apocalyptic world. Make it emotional and colorful.",
        "Two friends discover a magical portal in their school library. Adventure awaits!",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Test {i}/3")
        print("=" * 80)

        result = workflow.process_comic_request(prompt)
        print(result)

        print("\n" + "=" * 80)
        input("Press Enter to continue to next test...")


if __name__ == "__main__":
    main()
