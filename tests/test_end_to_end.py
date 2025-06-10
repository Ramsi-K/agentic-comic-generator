"""
End-to-end test suite for the multi-agent comic generation system.
Tests complete workflow including:
1. LLM & Tool Integration
2. Error Handling
3. Content Generation
4. Memory & Session Management
5. Response Validation
"""

import os
import json
import pytest
import asyncio
import shutil  # <-- Add this for cleanup
from pathlib import Path
from typing import Dict, Any, List

from services.session_id_generator import SessionIdGenerator
from agents.brown_workflow import BrownWorkflow, create_brown_workflow
from agents.bayko_workflow import BaykoWorkflow, create_agent_bayko


async def validate_session_artifacts(session_id: str) -> Dict[str, Any]:
    """Validate session artifacts and data persistence."""
    session_dir = Path(f"storyboard/{session_id}")
    artifacts = {
        "exists": session_dir.exists(),
        "llm_data": [],
        "agent_data": [],
        "output": [],
    }

    if artifacts["exists"]:
        # Check LLM data
        llm_dir = session_dir / "llm_data"
        if llm_dir.exists():
            artifacts["llm_data"] = list(llm_dir.glob("*.json"))

        # Check agent data
        agents_dir = session_dir / "agents"
        if agents_dir.exists():
            artifacts["agent_data"] = list(agents_dir.glob("*.json"))

        # Check outputs
        output_dir = session_dir / "output"
        if output_dir.exists():
            artifacts["output"] = list(output_dir.glob("*.*"))

    return artifacts


def test_comic_generation_flow():
    """Test complete comic generation flow with multiple test cases"""
    session_id = SessionIdGenerator.create_session_id("e2e_test")
    brown_workflow = create_brown_workflow(
        max_iterations=3, openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Test cases to cover different scenarios
    test_cases = [
        {
            "prompt": "A moody K-pop idol finds a puppy on the street. It changes everything.",
            "style": "Studio Ghibli",
            "panels": 4,
        },
        {
            "prompt": "A robot learns to paint in an art studio. Show emotional growth.",
            "style": "watercolor",
            "panels": 3,
        },
        # Test error case
        {
            "prompt": "",  # Empty prompt should trigger error
            "style": "invalid_style",
            "panels": -1,
            "expected_error": True,
        },
    ]

    successes = 0
    failures = 0

    for case in test_cases:
        print(f"\n🧪 Testing prompt: {case['prompt']}")
        print("=" * 80)

        try:
            # Handle error test cases
            if case.get("expected_error", False):
                enhanced_prompt = (
                    f"{case['prompt']} Style preference: {case['style']}"
                )
                result = asyncio.run(
                    brown_workflow.process_comic_request_async(enhanced_prompt)
                )
                assert (
                    result["status"] == "error"
                ), "Expected error status for invalid input"
                print("✅ Error handling working as expected")
                successes += 1
                continue

            # Regular test case
            enhanced_prompt = (
                f"{case['prompt']} Style preference: {case['style']}"
            )
            result = asyncio.run(
                brown_workflow.process_comic_request_async(enhanced_prompt)
            )

            # Print error details if status is error
            if result.get("status") == "error":
                print(f"❌ Error result: {json.dumps(result, indent=2)}")

            # Validate result structure
            assert result is not None, "Workflow returned None"
            assert "status" in result, "Missing status in result"
            assert (
                result["status"] == "success"
            ), f"Failed with status: {result['status']}"
            assert "bayko_response" in result, "Missing Bayko response"

            # Validate Bayko response
            bayko_data = result["bayko_response"]
            assert "panels" in bayko_data, "Missing panels in Bayko response"
            panels = bayko_data["panels"]

            # Validate panel count and content
            assert len(panels) > 0, "No panels generated"
            assert (
                len(panels) == case["panels"]
            ), f"Expected {case['panels']} panels, got {len(panels)}"

            for panel in panels:
                assert "description" in panel, "Panel missing description"

            # Verify session artifacts (just check session dir exists)
            artifacts = asyncio.run(validate_session_artifacts(session_id))
            assert artifacts["exists"], "Session directory not created"

            print(f"✅ Test case passed: {case['prompt'][:30]}...")
            successes += 1

        except AssertionError as e:
            print(f"❌ Test failed: {str(e)}")
            failures += 1
        except Exception as e:
            if not case.get("expected_error", False):
                print(f"❌ Unexpected error: {str(e)}")
                failures += 1
    print("\n" + "=" * 80)
    print("🎯 Test Summary")
    print(f"✅ Successful tests: {successes}")
    print(f"❌ Failed tests: {failures}")
    print(f"📊 Success rate: {(successes/(successes+failures))*100:.1f}%")
    try:
        # Final assertions
        assert successes > 0, "No test cases passed"
        assert failures == 0, "Some test cases failed"
    except AssertionError as e:
        print(f"❌ Test assertions failed: {str(e)}")
        raise
    finally:
        # Always cleanup and reset, even if tests fail
        try:
            test_dir = Path(f"storyboard/{session_id}")
            if test_dir.exists():
                shutil.rmtree(test_dir)
                print(f"✨ Cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"⚠️ Could not clean up test directory: {str(e)}")

        # Reset session state
        try:
            brown_workflow.reset()
            print("✨ Reset workflow state")
        except Exception as e:
            print(f"⚠️ Could not reset workflow state: {str(e)}")


def main():
    """Run all end-to-end tests"""
    print("\n🤖 Running Comic Generator End-to-End Tests")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return 1

    try:
        # Run comic generation flow test
        comic_flow_success = test_comic_generation_flow()
        print(
            "✅ Comic generation flow tests passed"
            if comic_flow_success
            else "❌ Comic generation flow tests failed"
        )

        # Final summary
        if comic_flow_success:
            print("\n🎉 All tests completed successfully!")
            return 0
        else:
            print("\n⚠️ Some tests failed - see details above")
            return 1

    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {str(e)}")
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
