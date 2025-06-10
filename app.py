import gradio as gr
import time
import json
import os
from agents.brown_workflow import create_brown_workflow
from pathlib import Path


# Initialize workflow
workflow = create_brown_workflow(
    max_iterations=3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def comic_generator(prompt, style_preference, verbose=True):
    """Verbose comic generation: stream all agent/tool messages and reasoning"""
    chat = []
    tool_calls = []
    images = []
    progress = 0

    try:
        # Start message
        chat.append(("🧐🧐 Agent Brown: Starting comic generation..."))
        progress += 5
        yield chat, images, tool_calls, progress

        # Process the request and get full trace
        result = workflow.process_comic_request(
            f"{prompt} Style preference: {style_preference}"
        )
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
            chat.append((f"🛠️ Tool Output:\n{tool_msg}"))
            tool_calls.append("tool_call")
            progress = min(progress + 10, 95)
            yield chat, images, tool_calls, progress

        # Show error if any
        if "error" in response_data:
            chat.append((f"❌ Error: {response_data['error']}"))
            progress = 100
            yield chat, images, tool_calls, progress
            return

        # Show Bayko's panel generation
        if "bayko_response" in response_data:
            bayko_data = response_data["bayko_response"]
            panels = bayko_data.get("panels", [])
            progress_per_panel = 50 / max(len(panels), 1)
            for i, panel in enumerate(panels, 1):
                chat.append(
                    (f"🧸 Agent Bayko: Panel {i}: {panel.get('caption', '')}",)
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
                yield chat, images, tool_calls, progress
                time.sleep(0.2)

        # Show Brown's analysis and decision
        if "analysis" in response_data:
            chat.append(
                (f"🧐 Agent Brown Analysis: {response_data['analysis']}",)
            )
            tool_calls.append("analyze_bayko_output")
            progress = min(progress + 10, 99)
            yield chat, images, tool_calls, progress

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
            yield chat, images, tool_calls, progress

        # If verbose, show the full response_data for debugging
        if verbose:
            chat.append(
                (
                    BROWN_AVATAR,
                    f"[DEBUG] Full response: {json.dumps(response_data, indent=2)}",
                )
            )
            yield chat, images, tool_calls, progress

    except Exception as e:
        chat.append((BROWN_AVATAR, f"❌ Error during generation: {str(e)}"))
        progress = 100
        yield chat, images, tool_calls, progress


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
        ⚠️ **Warning:** This demo is subject to OpenAI's strict rate limits (3 requests/min for gpt-4o). You may experience long waits (20+ seconds) between steps. If you see a rate limit error, please wait and try again, or use your own OpenAI API key above.
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
        tool_panel = gr.HighlightedText(
            label="Tool Calls",
            show_legend=True,
            color_map={
                "validate_input": "blue",
                "generate_panel_content": "orange",
                "analyze_bayko_output": "green",
                "final_decision": "purple",
                "tool_call": "gray",
            },
        )
        progress_bar = gr.Slider(
            minimum=0, maximum=100, value=0, step=1, label="Progress (%)"
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
        # Re-create the workflow with the user key
        workflow = create_brown_workflow(
            max_iterations=3,
            openai_api_key=key,
        )
        for chat, images, tool_calls, progress in comic_generator(
            prompt, style, verbose=True
        ):
            chat_display = [(msg[0], msg[1]) for msg in chat]
            tool_display = [(call, call) for call in tool_calls]
            yield chat_display, images, tool_display, progress

    submit_button.click(
        stream_comic,
        inputs=[openai_key_box, user_input, style_dropdown],
        outputs=[chat_window, image_gallery, tool_panel, progress_bar],
    )

    gr.Markdown(
        """
        ---
        <center>
        <b>Built with 🦙 LlamaIndex, Modal Labs, and Gradio for the Hugging Face Hackathon!</b>
        <br>Using GPT-4V for intelligent comic analysis
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
