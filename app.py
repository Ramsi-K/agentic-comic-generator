"""
🚧 PROJECT STATUS DISCLAIMER 🚧

This agentic comic generator project demonstrates a multi-agent workflow architecture
but is NOT fully functional due to several technical limitations:

KNOWN LIMITATIONS:
1. 🔴 OpenAI API Rate Limits: Severe restrictions on GPT-4 usage (3 requests/min)
2. 🔴 Modal .remote() Integration: Gradio doesn't support Modal's serverless functions properly
3. 🔴 Deployment Constraints: Full pipeline requires Modal infrastructure not available in this demo

WHAT WORKS:
✅ Individual agent components (Brown & Bayko) function correctly
✅ Agent-to-agent communication protocols are implemented
✅ LlamaIndex workflow architecture is properly structured
✅ UI components and chat interface work as intended
✅ Tool calling and response handling mechanisms are functional

WHAT'S INCOMPLETE:
❌ End-to-end comic generation pipeline
❌ Image generation via Modal serverless functions
❌ Full multi-iteration refinement workflow
❌ Production-ready error handling and recovery

This represents a proof-of-concept implementation showcasing the architectural
approach for multi-agent comic generation. The individual components communicate
effectively, but the complete workflow requires infrastructure beyond the current
deployment environment's capabilities.

For a fully functional version, deploy with proper Modal credentials and
infrastructure setup as outlined in the technical specifications.
"""

import gradio as gr
import time
import json
import os
import sys
import io
import logging
import threading
import queue
from contextlib import redirect_stdout, redirect_stderr
from agents.brown_workflow import create_brown_workflow
from pathlib import Path


# Initialize workflow
workflow = create_brown_workflow(
    max_iterations=3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def colorize_message(message, msg_type="default"):
    """Add color coding to messages based on type"""
    colors = {
        "agent_brown": "#2E86AB",  # Blue for Agent Brown
        "agent_bayko": "#A23B72",  # Purple for Agent Bayko
        "tool_output": "#F18F01",  # Orange for tool outputs
        "error": "#C73E1D",  # Red for errors
        "success": "#2D5016",  # Green for success
        "log": "#6C757D",  # Gray for logs
        "analysis": "#8E44AD",  # Purple for analysis
        "default": "#212529",  # Dark gray default
    }

    color = colors.get(msg_type, colors["default"])
    return f'<span style="color: {color}; font-weight: bold;">{message}</span>'


class LogCapture(io.StringIO):
    """Capture all print statements and logs for Gradio display"""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.original_stdout = sys.stdout

    def write(self, text):
        if text.strip():  # Only capture non-empty lines
            self.log_queue.put(text.strip())
        # Also write to original stdout so terminal still shows logs
        self.original_stdout.write(text)
        return len(text)

    def flush(self):
        self.original_stdout.flush()


def comic_generator(prompt, style_preference, verbose=True):
    """Verbose comic generation: stream all agent/tool messages and reasoning"""
    chat = []
    tool_calls = []
    images = []
    progress = 0

    # Create log queue and capture
    log_queue = queue.Queue()
    log_capture = LogCapture(log_queue)

    try:
        # Start message
        chat.append(
            (
                colorize_message(
                    "🧐 Agent Brown: Starting comic generation...",
                    "agent_brown",
                ),
            )
        )
        progress += 5
        yield chat, images

        # Start workflow in a separate thread to capture logs in real-time
        def run_workflow():
            with redirect_stdout(log_capture):
                return workflow.process_comic_request(
                    f"{prompt} Style preference: {style_preference}"
                )

        # Run workflow in thread
        workflow_thread = threading.Thread(
            target=lambda: setattr(run_workflow, "result", run_workflow())
        )
        workflow_thread.start()

        # Stream logs while workflow is running
        while workflow_thread.is_alive():
            try:
                # Get logs from queue with timeout
                log_message = log_queue.get(timeout=0.1)
                chat.append((colorize_message(f"📝 {log_message}", "log"),))
                progress = min(progress + 2, 90)
                yield chat, images
            except queue.Empty:
                continue

        # Wait for thread to complete
        workflow_thread.join()

        # Get any remaining logs
        while not log_queue.empty():
            try:
                log_message = log_queue.get_nowait()
                chat.append((colorize_message(f"📝 {log_message}", "log"),))
                yield chat, images
            except queue.Empty:
                break

        # Get the result
        result = getattr(run_workflow, "result", None)
        if not result:
            chat.append(("❌ No result from workflow",))
            yield chat, images
            return

        response_data = (
            json.loads(result) if isinstance(result, str) else result
        )

        # Show all tool outputs and chat history if available
        tool_outputs = response_data.get("tool_outputs", [])
        for tool_output in tool_outputs:
            # Try to pretty-print tool output JSON if possible
            try:
                tool_json = json.loads(tool_output)
                tool_msg = json.dumps(tool_json, indent=2)
            except Exception:
                tool_msg = str(tool_output)
            chat.append(
                (
                    colorize_message(
                        f"🛠️ Tool Output:\n{tool_msg}", "tool_output"
                    ),
                )
            )
            tool_calls.append("tool_call")
            progress = min(progress + 10, 95)
            yield chat, images

        # Show error if any
        if "error" in response_data:
            chat.append(
                (
                    colorize_message(
                        f"❌ Error: {response_data['error']}", "error"
                    ),
                )
            )
            progress = 100
            yield chat, images
            return

        # Show Bayko's panel generation
        if "bayko_response" in response_data:
            bayko_data = response_data["bayko_response"]
            panels = bayko_data.get("panels", [])
            progress_per_panel = 50 / max(len(panels), 1)
            for i, panel in enumerate(panels, 1):
                chat.append(
                    (
                        colorize_message(
                            f"🧸 Agent Bayko: Panel {i}: {panel.get('caption', '')}",
                            "agent_bayko",
                        ),
                    )
                )
                tool_calls.append("generate_panel_content")
                # Show image if available
                if "image_path" in panel:
                    img_path = Path(panel["image_path"])
                    if img_path.exists():
                        images.append(str(img_path.absolute()))
                elif "image_url" in panel:
                    images.append(panel["image_url"])
                progress += progress_per_panel
                yield chat, images
                time.sleep(0.2)

        # Show Brown's analysis and decision
        if "analysis" in response_data:
            chat.append(
                (
                    colorize_message(
                        f"🧐 Agent Brown Analysis: {response_data['analysis']}",
                        "analysis",
                    ),
                )
            )
            tool_calls.append("analyze_bayko_output")
            progress = min(progress + 10, 99)
            yield chat, images

        # Final decision
        if "decision" in response_data:
            decision = response_data["decision"]
            if decision == "APPROVE":
                chat.append(
                    (
                        "✅ Agent Brown: Comic approved! All quality checks passed.",
                    )
                )
            elif decision == "REFINE":
                chat.append(
                    (
                        "🔄 Agent Brown: Comic needs refinement. Starting another iteration...",
                    )
                )
            else:
                chat.append(
                    ("❌ Agent Brown: Comic rejected. Starting over...",)
                )
            tool_calls.append("final_decision")
            progress = 100
            yield chat, images

        # If verbose, show the full response_data for debugging
        if verbose:
            chat.append(
                (
                    f"[DEBUG] Full response: {json.dumps(response_data, indent=2)}",
                )
            )
            yield chat, images

    except Exception as e:
        chat.append(
            (
                colorize_message(
                    f"❌ Error during generation: {str(e)}", "error"
                ),
            )
        )
        progress = 100
        yield chat, images


def set_api_key(api_key):
    """Set the OpenAI API key as environment variable"""
    if not api_key or not api_key.strip():
        return colorize_message("❌ Please enter a valid API key", "error")

    if not api_key.startswith("sk-"):
        return colorize_message(
            "❌ Invalid API key format (should start with 'sk-')", "error"
        )

    # Set the environment variable
    os.environ["OPENAI_API_KEY"] = api_key.strip()

    # Update the global workflow with the new key
    global workflow
    workflow = create_brown_workflow(
        max_iterations=3,
        openai_api_key=api_key.strip(),
    )

    return colorize_message("✅ API key set successfully!", "success")


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🚧 PROJECT STATUS DISCLAIMER 🚧
        
        **This agentic comic generator is a PROOF-OF-CONCEPT demonstration with known limitations:**
        
        ### 🔴 Known Issues:
        - **OpenAI Rate Limits:** Severe restrictions (3 requests/min for GPT-4)
        - **Modal Integration:** Gradio doesn't support Modal .remote() functions properly
        - **Infrastructure:** Full pipeline requires Modal serverless deployment
        
        ### ✅ What Works:
        - Individual agent components (Brown & Bayko) function correctly
        - Agent-to-agent communication protocols are implemented
        - UI components and workflow architecture are functional
        
        ### ❌ What's Incomplete:
        - End-to-end comic generation pipeline
        - Image generation via Modal serverless functions
        - Full multi-iteration refinement workflow
        
        **This demonstrates the architectural approach but is NOT a complete working system.**
        """
    )

    gr.Markdown(
        """
        ⚠️ **Additional Warning:** This demo is subject to OpenAI's strict rate limits (3 requests/min for gpt-4o). You may experience long waits (20+ seconds) between steps. If you see a rate limit error, please wait and try again, or use your own OpenAI API key with higher limits.
        """
    )

    gr.Markdown(
        """
        # 🦙 Multi-Agent Comic Generator
        Enter a story prompt and let Brown & Bayko create a comic!
        Watch as the agents collaborate using LlamaIndex workflow and GPT-4V vision capabilities.
        """
    )

    with gr.Row():
        openai_key_box = gr.Textbox(
            label="Enter your OpenAI API Key (optional)",
            placeholder="sk-...",
            type="password",
            scale=4,
        )
        set_key_button = gr.Button("Set API Key 🔑", scale=1)
        key_status = gr.Textbox(
            label="Status",
            value="No API key set",
            interactive=False,
            scale=2,
        )

    with gr.Row():
        user_input = gr.Textbox(
            label="Enter your comic prompt",
            placeholder="A moody K-pop idol finds a puppy on the street...",
            scale=4,
        )
        style_dropdown = gr.Dropdown(
            ["Studio Ghibli", "Noir", "Manga", "Pixel Art"],
            label="Art Style",
            value="Studio Ghibli",
        )
        submit_button = gr.Button("Generate Comic 🎨")

    with gr.Row():
        chat_window = gr.Chatbot(
            label="Agent Conversation",
            bubble_full_width=False,
            show_copy_button=True,
            height=350,
        )

    with gr.Row():
        image_gallery = gr.Gallery(
            label="Comic Panels",
            columns=2,
            rows=2,
            height=400,
            object_fit="contain",
        )
        feedback = gr.Radio(
            ["👍 Love it!", "👎 Try again"],
            label="How's the comic?",
            value=None,
        )

    def stream_comic(openai_key, prompt, style):
        # Use user-provided OpenAI key if given, else fallback to env
        key = openai_key or os.getenv("OPENAI_API_KEY")

        # Set the API key as environment variable if provided
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        # Re-create the workflow with the user key
        workflow = create_brown_workflow(
            max_iterations=3,
            openai_api_key=key,
        )
        for chat, images in comic_generator(prompt, style, verbose=True):
            chat_display = []
            for msg in chat:
                if isinstance(msg, tuple) and len(msg) >= 1:
                    # If it's a single-element tuple, make it a proper chat message
                    if len(msg) == 1:
                        chat_display.append((msg[0], ""))
                    else:
                        chat_display.append(
                            (msg[0], msg[1] if len(msg) > 1 else "")
                        )
                else:
                    # Handle string messages
                    chat_display.append((str(msg), ""))
            yield chat_display, images

    submit_button.click(
        stream_comic,
        inputs=[openai_key_box, user_input, style_dropdown],
        outputs=[chat_window, image_gallery],
    )

    set_key_button.click(
        set_api_key,
        inputs=[openai_key_box],
        outputs=[key_status],
    )

    gr.Markdown(
        """
        ---
        <center>
        <b>Built with 🦙 LlamaIndex, Modal Labs, MistralAI and Gradio for the Hugging Face Hackathon!</b>
        <br>Using GPT-4 for intelligent comic analysis
        </center>
        """
    )

    gr.Markdown(
        """
        **🚀 Want the real version?**
        1. Clone this Space
        2. Set up Modal credentials 
        3. Deploy with actual Modal functions
        4. Enjoy serverless AI magic!
        
        *This demo shows the UI and MCP structure without Modal execution.*
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
