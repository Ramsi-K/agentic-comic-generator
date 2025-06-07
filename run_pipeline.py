"""
Agentic Comic Generator - Main Pipeline
Hackathon demo showcasing Agent Brown with LlamaIndex ReActAgent
"""

import os
import asyncio
from agents.brown_workflow import create_brown_workflow


def main():
    """
    Main pipeline for the Agentic Comic Generator
    Demonstrates Agent Brown using LlamaIndex ReActAgent for hackathon
    """

    print("🎨 Agentic Comic Generator - Hackathon Demo")
    print("🏆 Powered by LlamaIndex ReActAgent")
    print("=" * 60)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    # Create Brown workflow
    print("🤖 Initializing Agent Brown MultiModal ReAct Agent...")
    workflow = create_brown_workflow(max_iterations=3)
    print("✅ Agent Brown ready!")

    # Example prompts for demo
    demo_prompts = [
        {
            "title": "K-pop Idol & Puppy Story",
            "prompt": "A moody K-pop idol finds a puppy on the street. It changes everything. Use Studio Ghibli style with soft colors and 4 panels.",
        },
        {
            "title": "Robot Artist Story",
            "prompt": "A robot learns to paint in a post-apocalyptic world. Make it emotional and colorful with manga style.",
        },
        {
            "title": "Magical Portal Adventure",
            "prompt": "Two friends discover a magical portal in their school library. Adventure awaits! Use whimsical style with 6 panels.",
        },
    ]

    print(f"\n📚 Available Demo Stories ({len(demo_prompts)} options):")
    for i, story in enumerate(demo_prompts, 1):
        print(f"  {i}. {story['title']}")

    print("\n" + "=" * 60)

    # Interactive mode
    while True:
        print("\n🎯 Choose an option:")
        print("1-3: Run demo story")
        print("4: Enter custom prompt")
        print("q: Quit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == "q":
            print("👋 Thanks for trying the Agentic Comic Generator!")
            break

        elif choice in ["1", "2", "3"]:
            story_idx = int(choice) - 1
            story = demo_prompts[story_idx]

            print(f"\n🎬 Running Demo: {story['title']}")
            print("=" * 60)

            # Process the story
            result = workflow.process_comic_request(story["prompt"])
            print(result)

        elif choice == "4":
            print("\n✏️ Enter your custom story prompt:")
            custom_prompt = input("Prompt: ").strip()

            if custom_prompt:
                print(f"\n🎬 Processing Custom Story")
                print("=" * 60)

                result = workflow.process_comic_request(custom_prompt)
                print(result)
            else:
                print("❌ Empty prompt. Please try again.")

        else:
            print("❌ Invalid choice. Please try again.")

        print("\n" + "=" * 60)


async def async_demo():
    """
    Async demo version for testing async capabilities
    """
    print("🎨 Agentic Comic Generator - Async Demo")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    # Create workflow
    workflow = create_brown_workflow(max_iterations=3)

    # Test prompt
    prompt = "A moody K-pop idol finds a puppy on the street. It changes everything. Use Studio Ghibli style."

    print("🔄 Processing async request...")
    result = await workflow.process_comic_request_async(prompt)
    print(result)


def quick_test():
    """
    Quick test function for development
    """
    print("🧪 Quick Test - Agent Brown ReAct Demo")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    # Create workflow
    workflow = create_brown_workflow(max_iterations=3)

    # Test prompt
    test_prompt = "A robot learns to paint. Make it emotional with 3 panels."

    print(f"📝 Test Prompt: {test_prompt}")
    print("\n🔄 Processing...")

    result = workflow.process_comic_request(test_prompt)
    print(result)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
        elif sys.argv[1] == "async":
            asyncio.run(async_demo())
        else:
            print("Usage: python run_pipeline.py [test|async]")
    else:
        main()
