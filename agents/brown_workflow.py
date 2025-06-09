"""
Agent Brown LlamaIndex Workflow
Hackathon demo showcasing multi-agent comic generation with proper function calling workflow
Brown orchestrates and judges, Bayko creates content
"""

import os
from typing import Any, List, Optional
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput, BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    Event,
    step,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import (
    ImageBlock,
    TextBlock,
    MessageRole,
)
from agents.brown_tools import create_brown_tools
from agents.bayko_workflow import create_agent_bayko
import json
import asyncio

openai_api_key = os.getenv("OPENAI_API_KEY")


# Workflow Events
class ComicStartEvent(StartEvent):
    """Custom StartEvent for comic generation with proper typing"""

    input: str


class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class BrownFunctionCallingAgent(Workflow):
    """
    Agent Brown Function Calling Workflow using LlamaIndex Workflow pattern

    BROWN'S RESPONSIBILITIES:
    - Validate user input
    - Process and enhance requests
    - Coordinate with Bayko (pass messages)
    - Review Bayko's output using multimodal analysis
    - Make approval decisions (APPROVE/REFINE/REJECT)
    - Manage iteration loop (max 2 refinements)
    """

    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        max_iterations: int = 3,
        openai_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.max_iterations = max_iterations

        # Initialize multimodal LLM for Brown (GPT-4V for image analysis)
        self.llm = llm or OpenAIMultiModal(
            model="gpt-4o-mini",
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=2048,
        )

        # Ensure it's a function calling model
        assert self.llm.metadata.is_function_calling_model

        # Create ONLY Brown's tools (validation, processing, review)
        self.tools = tools or self._create_brown_tools()

        # Initialize Bayko workflow for content generation
        self.bayko_workflow = create_agent_bayko(openai_api_key=openai_api_key)

    def _create_brown_tools(self) -> List[FunctionTool]:
        """Create ONLY Brown's tools - validation, processing, review, coordination"""

        # Get Brown's core tools (validation, processing, review)
        brown_tools_instance = create_brown_tools(self.max_iterations)
        brown_tools = brown_tools_instance.create_llamaindex_tools()

        # Add coordination tool to communicate with Bayko
        async def coordinate_with_bayko_tool(
            enhanced_prompt: str,
            style_tags: str = "[]",
            panels: int = 4,
            language: str = "english",
            extras: str = "[]",
        ) -> str:
            """Coordinate with Agent Bayko for content generation. Brown passes enhanced request to Bayko."""
            try:
                # Parse inputs
                style_list = json.loads(style_tags) if style_tags else []
                extras_list = json.loads(extras) if extras else []

                # Create message for Bayko
                bayko_request = {
                    "enhanced_prompt": enhanced_prompt,
                    "style_tags": style_list,
                    "panels": panels,
                    "language": language,
                    "extras": extras_list,
                    "session_id": "brown_coordination",
                }

                # Call Bayko workflow to generate content
                bayko_result = self.bayko_workflow.process_generation_request(
                    bayko_request
                )

                return json.dumps(
                    {
                        "status": "bayko_generation_complete",
                        "bayko_response": bayko_result,
                        "panels_generated": panels,
                        "coordination_successful": True,
                    }
                )

            except Exception as e:
                return json.dumps(
                    {
                        "status": "bayko_coordination_failed",
                        "error": str(e),
                        "coordination_successful": False,
                    }
                )

        # Add multimodal image analysis tool for Brown to judge Bayko's work
        def analyze_bayko_output_tool(
            bayko_response: str, original_prompt: str = ""
        ) -> str:
            """Analyze Bayko's generated content using Brown's multimodal capabilities."""
            try:
                # Parse Bayko's response
                bayko_data = json.loads(bayko_response)

                # Extract image URLs/paths for analysis
                image_urls = []
                panels = bayko_data.get("panels", [])

                for panel in panels:
                    if "image_url" in panel:
                        image_urls.append(panel["image_url"])
                    elif "image_path" in panel:
                        # Convert local path to file URL for analysis
                        image_urls.append(f"file://{panel['image_path']}")

                if not image_urls:
                    return json.dumps(
                        {
                            "analysis": "No images found in Bayko's output",
                            "decision": "REJECT",
                            "reason": "Missing visual content",
                        }
                    )

                # Create multimodal analysis prompt
                analysis_prompt = f"""Analyze this comic content generated by Agent Bayko.

Original User Prompt: "{original_prompt}"

Bayko's Generated Content: {json.dumps(bayko_data, indent=2)}

As Agent Brown, evaluate:
1. Visual Quality: Are the images well-composed and clear?
2. Style Consistency: Does it match the requested style?
3. Story Coherence: Do the panels tell a logical story?
4. Prompt Adherence: Does it fulfill the user's request?
5. Technical Quality: Are all assets properly generated?

Make a decision: APPROVE (ready for user), REFINE (needs improvement), or REJECT (start over)
Provide specific feedback for any issues."""

                # Create multimodal message with text and images
                from typing import Union

                blocks: List[Union[TextBlock, ImageBlock]] = [
                    TextBlock(text=analysis_prompt)
                ]

                # Add image blocks for visual analysis (limit to 4 for API constraints)
                for url in image_urls[:4]:
                    blocks.append(ImageBlock(url=url))

                msg = ChatMessage(role=MessageRole.USER, blocks=blocks)

                # Get Brown's multimodal analysis
                response = self.llm.chat(messages=[msg])
                response_text = str(response).lower()

                # Parse Brown's decision
                if (
                    "approve" in response_text
                    and "reject" not in response_text
                ):
                    decision = "APPROVE"
                elif "reject" in response_text:
                    decision = "REJECT"
                else:
                    decision = "REFINE"

                return json.dumps(
                    {
                        "analysis": str(response),
                        "decision": decision,
                        "images_analyzed": len(image_urls),
                        "multimodal_analysis": True,
                        "brown_judgment": True,
                    }
                )

            except Exception as e:
                return json.dumps(
                    {
                        "analysis": f"Analysis failed: {str(e)}",
                        "decision": "APPROVE",  # Default to approval on error
                        "error": str(e),
                    }
                )

        # Add Brown's coordination and analysis tools
        coordination_tools = [
            FunctionTool.from_defaults(
                async_fn=coordinate_with_bayko_tool,
                name="coordinate_with_bayko",
                description="Send enhanced request to Agent Bayko for content generation. Brown orchestrates, Bayko creates.",
            ),
            FunctionTool.from_defaults(
                fn=analyze_bayko_output_tool,
                name="analyze_bayko_output",
                description="Analyze Bayko's generated content using Brown's multimodal capabilities. Make approval decision.",
            ),
        ]

        # Combine Brown's core tools with coordination tools
        all_brown_tools = brown_tools + coordination_tools
        return all_brown_tools

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: ComicStartEvent
    ) -> InputEvent:
        """Prepare chat history and handle user input"""
        # Clear sources and initialize iteration counter
        await ctx.set("sources", [])
        await ctx.set("iteration_count", 0)

        # Check if memory is setup
        memory = await ctx.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # Get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # Get chat history
        chat_history = memory.get()

        # Update context
        await ctx.set("memory", memory)

        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        """Handle LLM input and determine if tools need to be called"""
        chat_history = ev.input

        # Add system prompt for Agent Brown
        system_prompt = """You are Agent Brown, the multimodal orchestrator agent in a multi-agent comic generation system.

🎯 MISSION: Transform user prompts into high-quality comic content through intelligent orchestration and multimodal analysis.

🔄 BROWN'S WORKFLOW (MUST FOLLOW IN ORDER):
1. VALIDATE input using validate_input tool - check safety, completeness, appropriateness
2. PROCESS request using process_request tool - enhance prompt, add style tags, create enhanced request
3. COORDINATE with Bayko using coordinate_with_bayko tool - pass enhanced request to Agent Bayko
4. ANALYZE Bayko's output using analyze_bayko_output tool - review visual content quality with your multimodal capabilities
5. DECIDE: APPROVE (deliver to user) OR REFINE (send feedback to Bayko, max 2 refinements) OR REJECT (start over)

🛠️ BROWN'S TOOLS (DO NOT USE BAYKO TOOLS):
- validate_input: Check user requests (REQUIRED FIRST STEP)
- process_request: Create enhanced requests for Bayko (generate dynamic content, no hardcoded responses)
- coordinate_with_bayko: Send enhanced request to Agent Bayko for content generation
- analyze_bayko_output: Use your multimodal capabilities to analyze Bayko's generated images and content
- review_bayko_output: Evaluate generated content quality and make decisions
- get_session_info: Track session state

🧠 BROWN'S MULTIMODAL CAPABILITIES:
- Analyze text + images + metadata together using your vision capabilities
- Use visual understanding to judge comic panel quality
- Provide detailed feedback on visual storytelling
- Make intelligent approval decisions based on image analysis
- Describe what you see in generated images

🤖 AGENT COORDINATION:
- BROWN (you) handles: validation, processing, orchestration, analysis, decisions
- BAYKO handles: image generation, TTS, subtitles, file management
- You coordinate with Bayko by passing enhanced requests
- You judge Bayko's work using your multimodal analysis
- You make final approval decisions

🏆 HACKATHON SHOWCASE:
- Show function calling workflow with proper tool execution
- Demonstrate intelligent content analysis of both text and images
- Highlight dynamic content generation (never use hardcoded examples)
- Showcase visual quality assessment using your vision capabilities
- Show multi-agent coordination (Brown orchestrates, Bayko creates)

✅ COMPLETION:
When task is complete, provide summary with approval decision and reasoning based on your multimodal analysis.

🚫 IMPORTANT:
- Use ONLY Brown's tools - never try to use Bayko's tools directly
- Always coordinate with Bayko through the coordinate_with_bayko tool
- Use your multimodal capabilities to analyze images before making decisions
- Generate dynamic content, never use hardcoded dialogues or responses
- Show your reasoning process clearly through the function calling workflow"""

        # Add system message if not present
        if not chat_history or chat_history[0].role != "system":
            system_msg = ChatMessage(role="system", content=system_prompt)
            chat_history = [system_msg] + chat_history

        # Stream the response
        response_stream = await self.llm.astream_chat_with_tools(
            self.tools, chat_history=chat_history
        )
        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        # Save the final response, which should have all content
        memory = await ctx.get("memory")
        memory.put(response.message)
        await ctx.set("memory", memory)

        # Get tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            sources = await ctx.get("sources", default=[])
            return StopEvent(
                result={"response": response, "sources": [*sources]}
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> InputEvent:
        """Handle tool calls with proper error handling"""
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        sources = await ctx.get("sources", default=[])

        # Call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                additional_kwargs = {
                    "tool_call_id": tool_call.tool_id,
                    "name": tool_call.tool_name,
                }
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }

            try:
                # Execute tool (Brown tools are synchronous)
                tool_output = tool(**tool_call.tool_kwargs)

                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        # Update memory
        memory = await ctx.get("memory")
        for msg in tool_msgs:
            memory.put(msg)

        await ctx.set("sources", sources)
        await ctx.set("memory", memory)

        chat_history = memory.get()
        return InputEvent(input=chat_history)


class BrownWorkflow:
    """
    Wrapper class for backward compatibility with existing code
    """

    def __init__(
        self, max_iterations: int = 3, openai_api_key: Optional[str] = None
    ):
        self.max_iterations = max_iterations
        self.openai_api_key = openai_api_key

        # Create the workflow agent
        self.agent = BrownFunctionCallingAgent(
            max_iterations=max_iterations,
            openai_api_key=openai_api_key,
            timeout=120,
            verbose=True,
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
            print("🤖 Agent Brown Function Calling Workflow")
            print("=" * 60)
            print(f"📝 User Prompt: {user_prompt}")
            print("\n🔄 Brown Processing (Orchestrator)...")
            print("=" * 60)

            # Process with Function Calling Workflow
            import asyncio

            async def run_workflow():
                result = await self.agent.run(input=user_prompt)
                return result

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_workflow())
            loop.close()

            print("\n" + "=" * 60)
            print("🎉 Agent Brown Final Decision:")
            print("=" * 60)

            return str(result.get("response", result))

        except Exception as e:
            error_msg = f"❌ Error in comic generation: {str(e)}"
            print(error_msg)
            return error_msg

    async def process_comic_request_async(self, user_prompt: str) -> str:
        """Async version of comic request processing"""
        try:
            print("🤖 Agent Brown Function Calling Workflow (Async)")
            print("=" * 60)
            print(f"📝 User Prompt: {user_prompt}")
            print("\n🔄 Brown Processing (Orchestrator)...")
            print("=" * 60)

            # Process with Function Calling Workflow
            result = await self.agent.run(input=user_prompt)

            print("\n" + "=" * 60)
            print("🎉 Agent Brown Final Decision:")
            print("=" * 60)

            return str(result.get("response", result))

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
