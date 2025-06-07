#!/usr/bin/env python3
"""
Test script for the integrated memory and evaluation system
"""

import asyncio
from api.agents.brown import AgentBrown, StoryboardRequest
from api.agents.bayko import AgentBayko


async def test_integration():
    """Test the integrated memory and evaluation system"""

    print("🚀 Testing Bayko & Brown Integration")
    print("=" * 50)

    # Create agents
    brown = AgentBrown(max_iterations=3)
    bayko = AgentBayko()

    # Create test request
    request = StoryboardRequest(
        prompt="A cat discovers a magical portal in the garden",
        style_preference="studio_ghibli",
        panels=3,
        language="english",
        extras=["narration", "subtitles"],
    )

    print(f"📝 User prompt: {request.prompt}")
    print(f"🎨 Style: {request.style_preference}")
    print(f"📊 Panels: {request.panels}")
    print()

    # Step 1: Brown processes the request
    print("🤖 Brown processing request...")
    brown_message = brown.process_request(request)

    if brown_message.message_type == "validation_error":
        print(f"❌ Validation failed: {brown_message.payload}")
        return

    print(f"✅ Brown created request for Bayko")
    print(f"🧠 Brown memory size: {brown.get_session_info()['memory_size']}")
    print()

    # Step 2: Bayko generates content
    print("🎨 Bayko generating content...")
    bayko_result = await bayko.process_generation_request(
        brown_message.to_dict()
    )

    print(f"✅ Bayko generated {len(bayko_result.panels)} panels")
    print(f"⏱️  Total time: {bayko_result.total_time:.2f}s")
    print()

    # Step 3: Brown evaluates the result
    print("🔍 Brown evaluating Bayko's output...")
    review_result = brown.review_output(bayko_result.to_dict(), request)

    if review_result:
        print(f"📋 Review result: {review_result.message_type}")
        if review_result.message_type == "final_approval":
            print("🎉 Content approved!")
        elif review_result.message_type == "refinement_request":
            print("🔄 Refinement requested")
        else:
            print("❌ Content rejected")

    print()
    print("📊 Final Session Info:")
    session_info = brown.get_session_info()
    for key, value in session_info.items():
        print(f"  {key}: {value}")

    print()
    print("🧠 Memory Integration Test:")
    if brown.memory:
        history = brown.memory.get_history()
        print(f"  Brown memory entries: {len(history)}")
        for i, entry in enumerate(history[-3:], 1):  # Show last 3 entries
            print(f"    {i}. {entry.role}: {entry.content[:50]}...")

    if bayko.memory:
        history = bayko.memory.get_history()
        print(f"  Bayko memory entries: {len(history)}")
        for i, entry in enumerate(history[-3:], 1):  # Show last 3 entries
            print(f"    {i}. {entry.role}: {entry.content[:50]}...")


if __name__ == "__main__":
    asyncio.run(test_integration())
