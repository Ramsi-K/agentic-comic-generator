import modal
import os
import random

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

# Define the custom Modal image with Python execution requirements
image = modal.Image.debian_slim().pip_install(
    "mistralai",
    "nest_asyncio",
    "python-dotenv",
)

app = modal.App("fries-coder", image=image)


@app.function(
    secrets=[modal.Secret.from_name("mistral-api")],
    image=image,
    retries=3,
    timeout=300,
)
async def generate_and_run_script(prompt: str, session_id: str) -> dict:
    """
    Generate a Python script using Mistral Codestral and run it

    Args:
        prompt: Description of the script to generate

    Returns:
        dict with generated code, execution output, and status
    """
    import tempfile
    import subprocess
    from mistralai import Mistral

    try:
        # Initialize Mistral client
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

        # Create FIM prompt structure
        prefix = "# Write a short, funny Python script that:\n"
        code_start = "import random\nimport time\n\n"
        suffix = "\n\nif __name__ == '__main__':\n    main()"

        # Generate code using FIM
        response = client.fim.complete(
            model="codestral-latest",
            prompt=f"{prefix}{prompt}\n{code_start}",
            suffix=suffix,
            temperature=0.7,
            top_p=1,
        )

        generated_code = (
            f"{code_start}{response.choices[0].message.content}{suffix}"
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(generated_code)
            temp_path = f.name

        try:
            # Run the generated script
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=10,  # 10 second timeout
            )
            output = result.stdout
            error = None
            status = "success"
        except subprocess.TimeoutExpired:
            output = "Script execution timed out after 10 seconds"
            error = "timeout"
            status = "timeout"
        except subprocess.CalledProcessError as e:
            output = e.stdout
            error = e.stderr
            status = "runtime_error"
        finally:
            # Cleanup
            os.unlink(temp_path)

        return {
            "code": generated_code,
            "output": output,
            "error": error,
            "status": status,
        }

    except Exception as e:
        return {
            "code": None,
            "output": None,
            "error": str(e),
            "status": "generation_error",
        }


# Example usage
@app.local_entrypoint()
def main(session_id = "test_session_fries"):  
    animal = random.choice(
        [
            "cat",
            "dog",
            "fish",
            "bird",,
            "giraffe",
            "turtle",
            "monkey",
            "rabbit",
            "puppy",
            "animal"
        ]
    )
    prompt = f"""
create a simple ASCII art of a {animal}.
Create ASCII art using these characters: _ - = ~ ^ \\\\ / ( ) [ ] {{ }} < > | . o O @ *
Draw the art line by line with print statements.
Write a short, funny Python script.
Use only basic Python features.
Add a joke or pun about fries in the script.
Make it light-hearted and fun.
End with a message about fries.
Make sure the script runs without errors.
"""

    try:
        print(f"\n🤖 Generating an ascii {animal} for you!")
        result = generate_and_run_script.remote(prompt, session_id)

        # print("\n📝 Generated Code:")
        # print("=" * 40)
        # print(result["code"])
        print("=" * 30)
        print("\n🎮 Code Output:")
        print("=" * 30)
        print("\n\n")
        print(result["output"])

        # print("\n" + "=" * 80)

        print("🍟    🍟     🍟")
        print("Golden crispy Python fries")
        print("Coming right up!")
        print()
        print("Haha. Just kidding.")

        if result["code"]:
            # Save the generated code locally
            script_file = f"storyboard/{session_id}/output/fries_for_you.py"
            os.makedirs(os.path.dirname(script_file), exist_ok=True)
            with open(script_file, "w") as f:
                f.write(result["code"])
            print("\nGo here to check out your actual custom code:")
            print(f"👉 Code saved to: {script_file}")
            print("\n\n\n")

        if result["error"]:
            print("\n❌ Error:")
            print("=" * 40)
            print(result["error"])

        if result["error"]:
            print("Looks like there was an error during execution.")
            print("Here are some extra fries to cheer you up!")
            print("🍟    🍟     🍟")
            print("   🍟     🍟    ")
            print("       🍟      ")
            print("Now with extra machine-learned crispiness.")

    except modal.exception.FunctionTimeoutError:
        print("⏰ Script execution timed out after 300 seconds and 3 tries!")
        print("Sorry but codestral is having a hard time drawing today.")
        print("Here's a timeout fry for you! 🍟")
        print("Here are some extra fries to cheer you up!")
        print("🍟    🍟     🍟")
        print("   🍟     🍟    ")
        print("       🍟      ")
        print("Now with extra machine-learned crispiness.")
