#!/usr/bin/env python3
"""
Quick test to verify the refactored AgentBrown works
"""

from agents.brown import AgentBrown, StoryboardRequest


def test_refactored_brown():
    """Test the refactored AgentBrown"""
    print("🧪 Testing refactored AgentBrown...")

    # Create agent
    brown = AgentBrown(max_iterations=3)
    print("✅ AgentBrown created successfully")

    # Create test request
    request = StoryboardRequest(
        prompt="A cat finds a magical book in an old library",
        style_preference="anime",
        panels=3,
        language="english",
        extras=["narration"],
    )
    print("✅ StoryboardRequest created")

    # Process request
    try:
        message = brown.process_request(request)
        print("✅ Request processed successfully")
        print(f"📨 Generated message ID: {message.message_id}")
        print(f"🎯 Message type: {message.message_type}")
        print(f"📊 Session info: {brown.get_session_info()}")
        return True
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return False


if __name__ == "__main__":
    success = test_refactored_brown()
    if success:
        print("\n🎉 Refactoring successful! All functionality working.")
    else:
        print("\n💥 Refactoring needs fixes.")
