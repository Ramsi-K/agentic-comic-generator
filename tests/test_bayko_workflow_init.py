#!/usr/bin/env python3
"""
Test suite for BaykoWorkflow initialization and session handling
"""

import os
import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add pytest-asyncio marker
pytestmark = pytest.mark.asyncio

from agents.bayko_workflow import create_agent_bayko, BaykoWorkflow

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def create_test_workflow(api_key: Optional[str] = None) -> BaykoWorkflow:
    """Helper to create workflow with specified API key"""
    return create_agent_bayko(openai_api_key=api_key)


async def test_workflow_initialization():
    """Test core workflow initialization"""
    print("\n🧪 Testing BaykoWorkflow Initialization")
    print("=" * 60)

    # Test 1: With valid API key
    if os.getenv("OPENAI_API_KEY"):
        print("\n1️⃣ Testing initialization with API key")
        workflow = create_test_workflow(os.getenv("OPENAI_API_KEY"))

        assert workflow.llm is not None, "LLM should be initialized"
        assert workflow.agent is not None, "ReActAgent should be initialized"
        assert (
            workflow.bayko_agent is not None
        ), "Bayko agent should be initialized"
        assert workflow.tools is not None, "Tools should be initialized"
        print("✅ Workflow initialized with LLM capabilities")

    # Test 2: Without API key
    print("\n2️⃣ Testing initialization without API key")
    workflow_no_llm = create_test_workflow(None)

    assert workflow_no_llm.llm is None, "LLM should not be initialized"
    assert (
        workflow_no_llm.agent is None
    ), "ReActAgent should not be initialized"
    assert (
        workflow_no_llm.bayko_agent is not None
    ), "Bayko agent should still be initialized"
    assert (
        workflow_no_llm.tools is not None
    ), "Tools should still be initialized"
    print("✅ Workflow initialized in fallback mode")


async def test_session_initialization():
    """Test session initialization and handling"""
    print("\n🧪 Testing Session Initialization")
    print("=" * 60)

    workflow = create_test_workflow(os.getenv("OPENAI_API_KEY"))

    # Test 1: Basic session initialization
    print("\n1️⃣ Testing basic session initialization")
    session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    workflow.initialize_session(session_id)

    assert (
        workflow.session_manager is not None
    ), "Session manager should be initialized"
    assert workflow.memory is not None, "Memory should be initialized"
    assert (
        workflow.message_factory is not None
    ), "Message factory should be initialized"
    print("✅ Session services initialized")

    # Test 2: Session with custom conversation ID
    print("\n2️⃣ Testing session with custom conversation ID")
    session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    conv_id = "custom_conv_001"
    workflow.initialize_session(session_id, conv_id)

    assert (
        workflow.session_manager is not None
    ), "Session manager should be initialized"
    assert workflow.memory is not None, "Memory should be initialized"
    print("✅ Session initialized with custom conversation ID")


async def test_generation_request():
    """Test generation request handling"""
    print("\n🧪 Testing Generation Request Processing")
    print("=" * 60)

    workflow = create_test_workflow(os.getenv("OPENAI_API_KEY"))
    session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    workflow.initialize_session(session_id)

    # Create test request
    test_request = {
        "prompt": "A curious robot exploring a garden for the first time",
        "original_prompt": "Robot discovers nature",
        "style_tags": ["whimsical", "soft_lighting", "watercolor"],
        "panels": 2,
        "session_id": session_id,
    }

    # Test 1: Process with LLM if available
    print("\n1️⃣ Testing generation request processing")
    result = workflow.process_generation_request(test_request)
    assert result is not None, "Should get a result"
    print("✅ Successfully processed generation request")

    # Test 2: Verify fallback works
    print("\n2️⃣ Testing fallback generation")
    workflow_no_llm = create_test_workflow(None)
    workflow_no_llm.initialize_session(session_id)

    fallback_result = workflow_no_llm.process_generation_request(test_request)
    assert fallback_result is not None, "Should get a fallback result"
    assert (
        "fallback" in fallback_result.lower()
    ), "Should indicate fallback mode"
    print("✅ Fallback generation works")


async def main():
    """Run all tests"""
    print("🚀 Starting BaykoWorkflow Tests")
    print("=" * 80)

    try:
        await test_workflow_initialization()
        await test_session_initialization()
        await test_generation_request()

        print("\n✨ All tests completed successfully!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
