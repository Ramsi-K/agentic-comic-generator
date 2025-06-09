import modal
import io
import os

# Define the Modal image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "diffusers",
    "transformers",
    "torch",
    "safetensors",
    "accelerate",
    "Pillow",
    "python-dotenv",  # Add this for .env support
)

# Load environment variables only when running locally (not in Modal's cloud)
if modal.is_local():
    from dotenv import load_dotenv

    load_dotenv()

    # Set Modal credentials from .env file
    modal_token_id = os.environ.get("MODAL_TOKEN_ID")
    modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")

    if modal_token_id and modal_token_secret:
        os.environ["MODAL_TOKEN_ID"] = modal_token_id
        os.environ["MODAL_TOKEN_SECRET"] = modal_token_secret


app = modal.App("comic-image-generator", image=image)


class ComicImageGenerator:
    def __init__(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to("cuda")
        self.torch = torch

    def generate_comic_panel(
        self,
        prompt: str,
        panel_id: int,
        session_id: str,
        steps: int = 1,
        seed: int = 42,
    ) -> tuple:
        import time

        generator = self.torch.manual_seed(seed)
        start = time.time()
        result = self.pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=0.0,
            width=512,
            height=512,
            output_type="pil",
        )
        duration = time.time() - start
        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return buf.read(), duration


@app.function(image=image, gpu="A10G")
def generate_comic_panel(
    prompt: str,
    panel_id: int,
    session_id: str,
    steps: int = 1,
    seed: int = 42,
) -> tuple:
    generator = ComicImageGenerator()
    return generator.generate_comic_panel(
        prompt, panel_id, session_id, steps, seed
    )


@app.local_entrypoint()
def main():
    prompt = "A K-pop idol walking through a rainy Seoul street, whimsical, soft lighting, watercolor style"
    panel_id = 1
    session_id = "test_session_modal_image_gen"
    steps = 1  # SDXL Turbo works best with 1 step
    seed = 42

    # with app.run():
    img_bytes, duration = generate_comic_panel.remote(
        prompt, panel_id, session_id, steps, seed
    )
    # Save to local directory
    out_dir = f"storyboard/{session_id}/content"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/panel_{panel_id}.png"
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    print("✅ Image generated at:", out_path)
    print("🕒 Time taken:", duration, "seconds")
