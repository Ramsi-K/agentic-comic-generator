import gradio as gr
import time

# Dummy avatars (replace with your own image files or URLs)
BROWN_AVATAR = "https://i.imgur.com/8Km9tLL.png"
BAYKO_AVATAR = "https://i.imgur.com/1XqgQ5F.png"


# Simulated backend for streaming
def comic_generator(prompt, fries, style):
    chat = []
    tool_calls = []
    images = []
    progress = 0

    # Brown validates
    chat.append((BROWN_AVATAR, "🟦 Agent Brown: Validating your prompt..."))
    tool_calls.append("validate_input")
    progress += 10
    yield chat, images, tool_calls, progress

    time.sleep(0.7)
    # Bayko generates panel 1
    chat.append((BAYKO_AVATAR, "🟧 Agent Bayko: Generating panel 1..."))
    tool_calls.append("generate_panel_content")
    images.append(
        "https://placehold.co/300x200?text=Panel+1"
    )  # Replace with real image URLs
    progress += 20
    yield chat, images, tool_calls, progress

    time.sleep(0.7)
    # Brown analyzes
    chat.append((BROWN_AVATAR, "🟦 Agent Brown: Analyzing image quality..."))
    tool_calls.append("analyze_bayko_output")
    progress += 20
    yield chat, images, tool_calls, progress

    time.sleep(0.7)
    # Bayko generates panel 2
    chat.append((BAYKO_AVATAR, "🟧 Agent Bayko: Generating panel 2..."))
    tool_calls.append("generate_panel_content")
    images.append("https://placehold.co/300x200?text=Panel+2")
    progress += 20
    yield chat, images, tool_calls, progress

    time.sleep(0.7)
    # Brown final decision
    chat.append(
        (
            BROWN_AVATAR,
            f"🟦 Agent Brown: Comic complete! ('🍟 Fries included.' if fries else 'No fries.')",
        )
    )
    tool_calls.append("final_decision")
    progress = 100
    yield chat, images, tool_calls, progress


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🦙 Multi-Agent Comic Generator
        Enter a story prompt and let Brown & Bayko create a comic!  
        Enjoy streaming thoughts, tool calls, and more.  
        """
    )

    with gr.Row():
        user_input = gr.Textbox(label="Enter your comic prompt", scale=4)
        fries_checkbox = gr.Checkbox(label="Do you want fries with that?")
        style_dropdown = gr.Dropdown(
            ["Studio Ghibli", "Noir", "Manga", "Pixel Art"],
            label="Art Style",
            value="Studio Ghibli",
        )
        submit_button = gr.Button("Generate Comic")

    with gr.Row():
        chat_window = gr.Chatbot(
            label="Agent Conversation",
            avatar_images=[BROWN_AVATAR, BAYKO_AVATAR],
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
            },
        )
        progress_bar = gr.Progress(label="Progress")

    with gr.Row():
        image_gallery = gr.Gallery(
            label="Comic Panels", columns=2, rows=2, height="auto"
        )
        feedback = gr.Radio(["👍", "👎"], label="Did you like the comic?")

    def stream_comic(prompt, fries, style):
        for chat, images, tool_calls, progress in comic_generator(
            prompt, fries, style
        ):
            # For chat, Gradio expects a list of (avatar, message) tuples
            chat_display = [(msg[0], msg[1]) for msg in chat]
            # For tool calls, highlight the most recent
            tool_display = [(call, call) for call in tool_calls]
            yield chat_display, images, tool_display, progress

    submit_button.click(
        stream_comic,
        inputs=[user_input, fries_checkbox, style_dropdown],
        outputs=[chat_window, image_gallery, tool_panel, progress_bar],
    )

    gr.Markdown(
        """
        ---
        <center>
        <b>Made with 🦙 LlamaIndex, Modal, and Gradio for Hackathon Glory!</b>
        </center>
        """
    )

demo.launch()
