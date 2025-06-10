"""
Demo: Agent Brown with LlamaIndex ReActAgent
Shows how the original AgentBrown is used with LlamaIndex tools for hackathon
"""

import os
from dotenv import load_dotenv, find_dotenv
from agents.brown_workflow import create_brown_workflow

load_dotenv(find_dotenv())

print("DEBUG: loading .env...")
load_dotenv()
print("DEBUG: key =", os.getenv("OPENAI_API_KEY"))


def main():
    """Demo the multimodal Brown agent using your original AgentBrown class"""

    print("🤖 Agent Brown + LlamaIndex ReActAgent Demo")
    print("=" * 60)
    print("✅ Using your original AgentBrown class with LlamaIndex tools")
    print("🏆 Hackathon demo showcasing ReAct reasoning")
    print("=" * 60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return

    # Create workflow (uses your original AgentBrown internally)
    print("🔧 Creating Brown workflow with LlamaIndex ReActAgent...")
    workflow = create_brown_workflow(max_iterations=3)
    print("✅ Ready!")

    # Demo prompt
    prompt = """A moody K-pop idol finds a puppy on the street. It changes everything. 
    Use Studio Ghibli style with soft colors and 4 panels."""

    print(f"\n📝 Demo Prompt:")
    print(f"'{prompt}'")
    print("\n🔄 Processing with Agent Brown ReActAgent...")
    print("(Watch for Thought → Action → Observation pattern)")
    print("=" * 60)

    # Process with visible reasoning
    result = workflow.process_comic_request(prompt)

    print("\n" + "=" * 60)
    print("🎉 Final Result:")
    print(result)
    print("=" * 60)
    print(
        "✅ Demo complete! Your original AgentBrown methods were used as tools."
    )


if __name__ == "__main__":
    main()
