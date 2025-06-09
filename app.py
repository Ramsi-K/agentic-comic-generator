import gradio as gr
from agents.brown_workflow import create_brown_workflow

workflow = create_brown_workflow()


def generate_comic(user_prompt, fries):
    # This is a placeholder for your streaming logic
    # You would yield tuples: (chat_history, images, tool_calls, progress, ...)
    # For demo, just return static data
    chat_history = [
        ("Agent Brown", "Validating your prompt..."),
        ("Agent Bayko", "Generating panel 1..."),
        ("Agent Brown", "Analyzing image quality..."),
    ]
    images = [
        "panel1.png",
        "panel2.png",
        "panel3.png",
        "panel4.png",
    ]  # Replace with real paths/URLs
    tool_calls = [
        "validate_input",
        "process_request",
        "coordinate_with_bayko",
        "analyze_bayko_output",
    ]
    progress = 100
    return chat_history, images, tool_calls, progress


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦙 Multi-Agent Comic Generator")
    gr.Markdown(
        "Enter a story prompt and let Brown & Bayko create a comic! Enjoy streaming thoughts, tool calls, and more."
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
            avatar_images=["brown_avatar.png", "bayko_avatar.png"],
        )
        tool_panel = gr.HighlightedText(label="Tool Calls")
        progress_bar = gr.Progress(label="Progress")

    with gr.Row():
        image_gallery = gr.Gallery(
            label="Comic Panels", columns=4, rows=1, height="auto"
        )
        feedback = gr.Radio(["👍", "👎"], label="Did you like the comic?")

    submit_button.click(
        generate_comic,
        inputs=[user_input, fries_checkbox],
        outputs=[chat_window, image_gallery, tool_panel, progress_bar],
    )

demo.launch()
