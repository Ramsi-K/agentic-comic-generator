"""
Agent Brown Workflow - Streamlined for hackathon demo
"""

import os
import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Sequence
from datetime import datetime
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from agents.brown import AgentBrown, StoryboardRequest
from agents.brown_tools import create_brown_tools
from agents.bayko_workflow import create_agent_bayko
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, BaseTool
from llama_index.core.llms import (
    ChatMessage,
    ImageBlock,
    TextBlock,
    MessageRole,
)
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import Field

from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools.types import ToolOutput
from llama_index.core.workflow import Context
from prompts.brown_workflow_system_prompt import BROWN_WORKFLOW_SYSTEM_PROMPT


# Global LLM throttle
_llm_lock = asyncio.Lock()
_last_llm_time = 0


async def throttle_llm():
    global _last_llm_time
    async with _llm_lock:
        now = time.time()
        wait = max(0, 21 - (now - _last_llm_time))
        if wait > 0:
            await asyncio.sleep(wait)
        _last_llm_time = time.time()


# Workflow Events
class InputEvent(Event):
    def __init__(self, input: list):
        super().__init__()
        self.input = input


class StreamEvent(Event):
    def __init__(self, delta: str):
        super().__init__()
        self.delta = delta


class ToolCallEvent(Event):
    def __init__(self, tool_calls: list):
        super().__init__()
        self.tool_calls = tool_calls


class FunctionOutputEvent(Event):
    def __init__(self, output):
        super().__init__()
        self.output = output


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
        llm: OpenAIMultiModal | None = None,
        tools: List[BaseTool] | None = None,
        max_iterations: int = 1,  # Force only one iteration
        openai_api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, timeout=timeout, **kwargs)

        self.max_iterations = 1  # Force only one iteration

        # Initialize multimodal LLM for Brown (GPT-4V for image analysis)
        self.llm = llm or OpenAIMultiModal(
            model="gpt-4o-mini",
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=2048,
            additional_kwargs={"tool_choice": "required"},
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
            """
            Send the enhanced comic request to Agent Bayko for actual comic content generation.
            This is the ONLY way to generate comic panels and story content.
            Always use this tool when the user prompt or workflow requires new comic content.

            Arguments:
                enhanced_prompt: The improved user prompt for the comic.
                style_tags: JSON list of style tags (e.g., '["manga", "noir"]').
                panels: Number of comic panels to generate.
                language: Language for the comic.
                extras: JSON list of extra instructions.

            Returns:
                JSON string with Bayko's response and status. Example:
                '{"status": "bayko_generation_complete", "bayko_response": {...}, ...}'
            """
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
                # If Bayko is async, use: bayko_result = await self.bayko_workflow.process_generation_request(bayko_request)
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
                print(f"[Brown] Bayko coordination failed: {e}")  # Debug log
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
                description="Send the enhanced comic request to Agent Bayko for actual comic content generation. This is the ONLY way to generate comic panels and story content. Always use this tool when the user prompt or workflow requires new comic content. Returns a JSON string with Bayko's response and status.",
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
        self, ctx: Context, ev: StartEvent
    ) -> InputEvent:
        """Prepare chat history and handle user input"""
        # Clear sources and initialize iteration counter
        await ctx.set("sources", [])
        await ctx.set("iteration_count", 0)

        # Store original user prompt in context
        await ctx.set("original_prompt", ev.input)

        # Check if memory is setup
        memory = await ctx.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # Add system message if not present
        if not memory.get() or memory.get()[0].role != "system":
            system_msg = ChatMessage(
                role="system", content=BROWN_WORKFLOW_SYSTEM_PROMPT
            )
            memory.put(system_msg)

        # Get user input and add to memory
        user_msg = ChatMessage(role="user", content=ev.input)
        memory.put(user_msg)

        # Update context with memory
        await ctx.set("memory", memory)

        # Return chat history
        return InputEvent(input=memory.get())

    @step
    async def enhance_and_send_to_bayko(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """
        Single step to enhance the user prompt and immediately send it to Bayko.
        """
        # Retrieve the original user prompt
        original_prompt = ev.input

        # Add system message for enhancement
        system_msg = ChatMessage(role="system", content=BROWN_WORKFLOW_SYSTEM_PROMPT)
        user_msg = ChatMessage(role="user", content=original_prompt)

        # Prepare chat history
        chat_history = [system_msg, user_msg]

        # Throttle LLM call
        await throttle_llm()

        # Enhance the prompt using a single LLM call
        response = await self.llm.chat(chat_history)
        enhanced_prompt = response.message.content

        # Log the enhanced prompt for debugging
        print(f"[DEBUG] Enhanced Prompt: {enhanced_prompt}")

        # Call Bayko directly with the enhanced prompt
        bayko_request = {
            "enhanced_prompt": enhanced_prompt,
            "style_tags": "[]",  # Default style tags
            "panels": 4,          # Default number of panels
            "language": "english",  # Default language
            "extras": "[]",     # Default extras
        }

        try:
            bayko_result = self.bayko_workflow.process_generation_request(bayko_request)

            # Log Bayko's response for debugging
            print(f"[DEBUG] Bayko Response: {bayko_result}")

            # Return a successful stop event
            return StopEvent(output={
                "status": "success",
                "bayko_response": bayko_result,
            })
        except Exception as e:
            # Log the error for debugging
            print(f"[ERROR] Bayko Coordination Failed: {e}")

            # Return a failure stop event
            return StopEvent(output={
                "status": "error",
                "error": str(e),
            })

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        """Handle LLM input and determine if tools need to be called"""
        import asyncio

        chat_history = ev.input

        # Add system prompt for Agent Brown
        system_prompt = BROWN_WORKFLOW_SYSTEM_PROMPT

        # Add system message if not present
        if not chat_history or chat_history[0].role != "system":
            system_msg = ChatMessage(role="system", content=system_prompt)
            chat_history = [system_msg] + chat_history

        await throttle_llm()
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
            response, error_on_no_tool_call=True
        )

        if not tool_calls:
            # If no tool call, error out immediately
            raise RuntimeError(
                "Agent Brown did not use the required tool. The workflow will stop."
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> InputEvent:
        """Handle tool calls with proper error handling (supports async tools)"""
        import inspect
        import asyncio

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
                # Check if tool is async and call appropriately
                if inspect.iscoroutinefunction(tool):
                    tool_output = await tool(**tool_call.tool_kwargs)
                else:
                    tool_output = tool(**tool_call.tool_kwargs)

                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=(
                            tool_output.content
                            if hasattr(tool_output, "content")
                            else str(tool_output)
                        ),
                        additional_kwargs=additional_kwargs,
                    )
                )
                # Throttle after each tool call
                await asyncio.sleep(21)
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
        self, max_iterations: int = 1, openai_api_key: Optional[str] = None
    ):
        self.max_iterations = 1
        self.openai_api_key = openai_api_key

        # Get Brown's tools
        brown_tools = create_brown_tools(max_iterations)
        self.tools: Sequence[BaseTool] = brown_tools.create_llamaindex_tools()

        # Initialize LLM
        self.llm = OpenAIMultiModal(
            model="gpt-4o",
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=2048,
            additional_kwargs={"tool_choice": "required"},
        )

    def reset(self):
        """Reset workflow state for a new session."""
        self.session_id = None
        # Reset agent instances
        brown_tools = create_brown_tools(self.max_iterations)
        self.tools = brown_tools.create_llamaindex_tools()
        # Reset LLM if needed
        self.llm = OpenAIMultiModal(
            model="gpt-4o",
            api_key=self.openai_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=2048,
            additional_kwargs={"tool_choice": "required"},
        )

    async def process_comic_request_async(
        self, user_prompt: str
    ) -> Dict[str, Any]:
        """Async version of comic request processing"""
        try:
            # Create workflow agent
            workflow = BrownFunctionCallingAgent(
                llm=self.llm,
                tools=list(self.tools),  # Convert Sequence to List
                max_iterations=self.max_iterations,
                openai_api_key=self.openai_api_key,
                timeout=None,
                verbose=True,
            )

            # Run workflow with user prompt as input (LlamaIndex pattern)
            result = await workflow.run(input=user_prompt)

            # Parse response
            response = result.get("response")
            if not response:
                return {
                    "status": "error",
                    "error": "No response from workflow",
                }

            # Extract relevant data
            tool_outputs = result.get("sources", [])
            bayko_outputs = [
                out
                for out in tool_outputs
                if "bayko_response" in str(out.content)
            ]

            if not bayko_outputs:
                return {
                    "status": "error",
                    "error": "No content generated by Bayko",
                }

            # Get last Bayko output (final version)
            try:
                final_output = json.loads(bayko_outputs[-1].content)
                bayko_response = final_output.get("bayko_response", {})

                return {
                    "status": "success",
                    "bayko_response": bayko_response,
                    "decision": final_output.get("decision", "APPROVE"),
                    "analysis": final_output.get("analysis", ""),
                    "tool_outputs": [str(out.content) for out in tool_outputs],
                }
            except json.JSONDecodeError as e:
                return {
                    "status": "error",
                    "error": f"Failed to parse Bayko response: {str(e)}",
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def process_comic_request(self, user_prompt: str) -> Dict[str, Any]:
        """Synchronous version that runs the async version"""
        import asyncio

        return asyncio.run(self.process_comic_request_async(user_prompt))


def create_brown_workflow(
    max_iterations: int = 1, openai_api_key: Optional[str] = None
) -> BrownWorkflow:
    """
    Factory function to create and initialize Brown workflow
    """
    return BrownWorkflow(
        max_iterations=max_iterations, openai_api_key=openai_api_key
    )


# def create_brown_workflow(
#     max_iterations: int = 3,
#     openai_api_key: Optional[str] = None,
# ) -> BrownFunctionCallingAgent:
#     """Create and initialize Brown workflow"""

#     workflow = BrownFunctionCallingAgent(
#         max_iterations=max_iterations,
#         openai_api_key=openai_api_key,
#     )
#     return workflow
