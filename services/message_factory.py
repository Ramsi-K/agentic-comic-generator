"""
Message Factory Service
Handles creation of AgentMessage objects with proper formatting and validation.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class MessageType(Enum):
    """Types of messages Agent Brown can send"""

    GENERATION_REQUEST = "generation_request"
    REFINEMENT_REQUEST = "refinement_request"
    VALIDATION_ERROR = "validation_error"
    FINAL_APPROVAL = "final_approval"


@dataclass
class AgentMessage:
    """Schema for inter-agent communication following tech_specs.md"""

    message_id: str
    timestamp: str
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class MessageFactory:
    """Factory for creating standardized AgentMessage objects"""

    def __init__(self, session_id: str, conversation_id: str):
        self.session_id = session_id
        self.conversation_id = conversation_id

    def create_generation_request(
        self,
        enhanced_prompt: str,
        original_prompt: str,
        dialogues: List[str],
        style_tags: List[str],
        panels: int,
        language: str,
        extras: List[str],
        style_config: Dict[str, Any],
        validation_score: float,
        iteration: int,
    ) -> AgentMessage:
        """Create a generation request message for Agent Bayko"""
        payload = {
            "prompt": enhanced_prompt,
            "original_prompt": original_prompt,
            "style_tags": style_tags,
            "panels": panels,
            "language": language,
            "extras": extras,
            "style_config": style_config,
            "generation_params": {
                "quality": "high",
                "aspect_ratio": "16:9",
                "panel_layout": "sequential",
            },
        }

        return AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_brown",
            recipient="agent_bayko",
            message_type=MessageType.GENERATION_REQUEST.value,
            payload=payload,
            context={
                "conversation_id": self.conversation_id,
                "session_id": self.session_id,
                "iteration": iteration,
                "previous_feedback": None,
                "validation_score": validation_score,
            },
        )

    def create_error_message(
        self, issues: List[str], suggestions: List[str]
    ) -> AgentMessage:
        """Create error message for validation failures"""
        return AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_brown",
            recipient="user_interface",
            message_type=MessageType.VALIDATION_ERROR.value,
            payload={
                "error": "Input validation failed",
                "issues": issues,
                "suggestions": suggestions,
            },
            context={
                "conversation_id": self.conversation_id or "error",
                "session_id": self.session_id or "error",
                "iteration": 0,
                "error_type": "validation",
            },
        )

    def create_rejection_message(
        self,
        bayko_response: Dict[str, Any],
        evaluation: Dict[str, Any],
        iteration: int,
    ) -> AgentMessage:
        """Create rejection message for auto-rejected content"""
        return AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_brown",
            recipient="user_interface",
            message_type=MessageType.VALIDATION_ERROR.value,
            payload={
                "error": "Content rejected",
                "reason": evaluation["reason"],
                "rejected_content": bayko_response,
                "auto_rejection": True,
            },
            context={
                "conversation_id": self.conversation_id,
                "session_id": self.session_id,
                "iteration": iteration,
                "rejection_type": "quality",
            },
        )

    def create_refinement_message(
        self,
        bayko_response: Dict[str, Any],
        feedback: Dict[str, Any],
        iteration: int,
    ) -> AgentMessage:
        """Create refinement request message"""
        return AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_brown",
            recipient="agent_bayko",
            message_type=MessageType.REFINEMENT_REQUEST.value,
            payload={
                "original_content": bayko_response,
                "feedback": feedback,
                "specific_improvements": feedback.get(
                    "improvement_suggestions", []
                ),
                "focus_areas": [
                    area
                    for area, score in [
                        ("adherence", feedback.get("adherence_score", 0)),
                        (
                            "style_consistency",
                            feedback.get("style_consistency", 0),
                        ),
                        ("narrative_flow", feedback.get("narrative_flow", 0)),
                        (
                            "technical_quality",
                            feedback.get("technical_quality", 0),
                        ),
                    ]
                    if score < 0.7
                ],
                "iteration": iteration,
            },
            context={
                "conversation_id": self.conversation_id,
                "session_id": self.session_id,
                "iteration": iteration,
                "previous_feedback": feedback,
                "refinement_reason": "Quality below threshold",
            },
        )

    def create_approval_message(
        self,
        bayko_response: Dict[str, Any],
        feedback: Dict[str, Any],
        iteration: int,
    ) -> AgentMessage:
        """Create final approval message"""
        return AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sender="agent_brown",
            recipient="user_interface",
            message_type=MessageType.FINAL_APPROVAL.value,
            payload={
                "approved_content": bayko_response,
                "final_feedback": feedback,
                "session_summary": {
                    "total_iterations": iteration,
                    "final_score": feedback.get("overall_score", 0),
                    "processing_complete": True,
                },
            },
            context={
                "conversation_id": self.conversation_id,
                "session_id": self.session_id,
                "iteration": iteration,
                "final_approval": True,
                "completion_timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )
