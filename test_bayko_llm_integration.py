#!/usr/bin/env python3
"""
Comprehensive test for Agent Bayko LLM integration
Tests both LLM and fallback modes with proper session management
"""

import os
import asyncio
import json
from pathlib import Path
from agents.bayko import create_agent_bayko

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


async def test_bayko_llm_integration():
    """Comprehensive test of Bayko LLM integration"""

    print("🧪 Testing Agent Bayko LLM Integration")
    print("=" * 60)

    # Test 1: Bayko without LLM (fallback mode)
    print("\n1️⃣ Testing Bayko without LLM (fallback mode)")
    print("-" * 40)
    bayko_fallback = create_agent_bayko(llm=None)

    # Test prompt generation fallback
    description = "A K-pop idol walking alone in the rain, feeling melancholy"
    style_tags = ["whimsical", "soft_lighting", "watercolor", "studio_ghibli"]
    mood = "melancholy"

    print(f"📝 Original description: {description}")
    print(f"🎨 Style tags: {style_tags}")
    print(f"😔 Mood: {mood}")

    result_fallback = bayko_fallback.generate_prompt_from_description(
        description, style_tags, mood
    )
    print(f"🔄 Fallback result: {result_fallback}")

    # Test feedback revision fallback
    feedback = {
        "improvement_suggestions": [
            "Improve visual style consistency",
            "Enhance emotional depth",
        ],
        "overall_score": 0.6,
        "reason": "Needs better style consistency and emotional impact",
    }
    focus_areas = ["style_consistency", "narrative_flow"]

    revised_fallback = bayko_fallback.revise_panel_description(
        description, feedback, focus_areas
    )
    print(f"🔄 Fallback revision: {revised_fallback}")

    # Test 2: Bayko with LLM (if available)
    if OpenAI and os.getenv("OPENAI_API_KEY"):
        print("\n2️⃣ Testing Bayko with LLM")
        print("-" * 40)
        try:
            llm = OpenAI()
            bayko_llm = create_agent_bayko(llm=llm)

            print(f"📝 Original description: {description}")
            print(f"🎨 Style tags: {style_tags}")
            print(f"😔 Mood: {mood}")

            # Test LLM prompt generation
            print("\n🤖 Testing LLM prompt generation...")
            result_llm = bayko_llm.generate_prompt_from_description(
                description, style_tags, mood
            )
            print(f"✨ LLM enhanced prompt: {result_llm}")

            # Test LLM feedback revision
            print("\n🤖 Testing LLM feedback revision...")
            print(f"📋 Feedback: {feedback}")
            print(f"🎯 Focus areas: {focus_areas}")

            revised_llm = bayko_llm.revise_panel_description(
                description, feedback, focus_areas
            )
            print(f"✨ LLM revised description: {revised_llm}")

            # Test session management and memory
            print("\n🧠 Testing session management...")
            session_info = bayko_llm.get_session_info()
            print(f"📊 Session info: {session_info}")

            # Check if LLM data was saved
            if (
                hasattr(bayko_llm, "current_session")
                and bayko_llm.current_session
            ):
                session_dir = Path(
                    f"storyboard/{bayko_llm.current_session}/llm_data"
                )
                if session_dir.exists():
                    llm_files = list(session_dir.glob("*.json"))
                    print(f"💾 LLM data files saved: {len(llm_files)}")
                    for file in llm_files:
                        print(f"   📄 {file.name}")
                        # Show content of first file
                        if file.name.startswith("generation_"):
                            with open(file, "r") as f:
                                data = json.load(f)
                                print(
                                    f"   📋 Content: {data.get('generated_prompt', 'N/A')[:100]}..."
                                )
                else:
                    print("⚠️  No LLM data directory found")

        except Exception as e:
            print(f"❌ LLM test failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
    else:
        print("\n2️⃣ Skipping LLM test")
        print("-" * 40)
        if not OpenAI:
            print("⚠️  OpenAI package not available")
        elif not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY not found in environment")
        else:
            print("⚠️  Unknown issue with OpenAI setup")

    # Test 3: Compare outputs
    print("\n3️⃣ Comparison Summary")
    print("-" * 40)
    print(f"📏 Fallback prompt length: {len(result_fallback)} chars")
    print(f"📏 Fallback revision length: {len(revised_fallback)} chars")

    if "result_llm" in locals():
        print(f"📏 LLM prompt length: {len(result_llm)} chars")
        print(f"📏 LLM revision length: {len(revised_llm)} chars")
        print(
            f"🔍 LLM enhancement factor: {len(result_llm) / len(result_fallback):.2f}x"
        )

    print("\n✅ Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_bayko_llm_integration())
