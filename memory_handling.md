# Memory Handling for Bayko & Brown

## Hackathon Implementation Guide

---

## 🧠 LlamaIndex Memory Integration

### Real Memory Class (Based on LlamaIndex Docs)

```python
# services/agent_memory.py
from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage

class AgentMemory:
    """Simple wrapper around LlamaIndex Memory for agent conversations"""

    def __init__(self, session_id: str, agent_name: str):
        self.session_id = session_id
        self.agent_name = agent_name

        # Use LlamaIndex Memory with session-specific ID
        self.memory = Memory.from_defaults(
            session_id=f"{session_id}_{agent_name}",
            token_limit=4000
        )

    def add_message(self, role: str, content: str):
        """Add a message to memory"""
        message = ChatMessage(role=role, content=content)
        self.memory.put_messages([message])

    def get_history(self):
        """Get conversation history"""
        return self.memory.get()

    def clear(self):
        """Clear memory for new session"""
        self.memory.reset()
```

### Integration with Existing Agents

**Update Brown's memory (api/agents/brown.py):**

```python
# Replace the LlamaIndexMemoryStub with real memory
from services.agent_memory import AgentMemory

class AgentBrown:
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.session_id = None
        self.iteration_count = 0

        # Real LlamaIndex memory
        self.memory = None  # Initialize when session starts

        # ... rest of existing code

    def process_request(self, request: StoryboardRequest):
        # Initialize memory for new session
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.memory = AgentMemory(self.session_id, "brown")

        # Log user request
        self.memory.add_message("user", request.prompt)

        # ... existing validation and processing logic

        # Log Brown's decision
        self.memory.add_message("assistant", f"Created generation request for Bayko")

        return message
```

**Update Bayko's memory (api/agents/bayko.py):**

```python
# Add memory to Bayko
from services.agent_memory import AgentMemory

class AgentBayko:
    def __init__(self):
        # ... existing initialization
        self.memory = None  # Initialize when processing starts

    async def process_generation_request(self, message: Dict[str, Any]):
        session_id = message.get("context", {}).get("session_id")
        self.memory = AgentMemory(session_id, "bayko")

        # Log received request
        self.memory.add_message("user", f"Received generation request: {message['payload']['prompt']}")

        # ... existing generation logic

        # Log completion
        self.memory.add_message("assistant", f"Generated {len(panels)} panels successfully")

        return result
```

### Optional: Sync with SQLite

```python
# services/memory_sync.py
from services.turn_memory import AgentMemory as SQLiteMemory
from services.agent_memory import AgentMemory as LlamaMemory

def sync_to_sqlite(llama_memory: LlamaMemory, sqlite_memory: SQLiteMemory):
    """Sync LlamaIndex memory to SQLite for persistence"""
    history = llama_memory.get_history()

    for message in history:
        sqlite_memory.add_message(
            session_id=llama_memory.session_id,
            agent_name=llama_memory.agent_name,
            content=message.content,
            step_type="message"
        )
```

---

## Evaluation Logic

### Basic Evaluator Class

```python
# services/simple_evaluator.py

class SimpleEvaluator:
    """Basic evaluation logic for Brown's decision making"""

    MAX_ATTEMPTS = 3  # Original + 2 revisions

    def __init__(self):
        self.attempt_count = 0

    def evaluate(self, bayko_output: dict, original_prompt: str) -> dict:
        """Evaluate Bayko's output and decide: approve, reject, or refine"""
        self.attempt_count += 1

        print(f"🔍 Brown evaluating attempt {self.attempt_count}/{self.MAX_ATTEMPTS}")

        # Rule 1: Auto-reject if dialogue in images
        if self._has_dialogue_in_images(bayko_output):
            return {
                "decision": "reject",
                "reason": "Images contain dialogue text - use subtitles instead",
                "final": True
            }

        # Rule 2: Auto-reject if story is incoherent
        if not self._is_story_coherent(bayko_output):
            return {
                "decision": "reject",
                "reason": "Story panels don't follow logical sequence",
                "final": True
            }

        # Rule 3: Force approve if max attempts reached
        if self.attempt_count >= self.MAX_ATTEMPTS:
            return {
                "decision": "approve",
                "reason": f"Max attempts ({self.MAX_ATTEMPTS}) reached - accepting current quality",
                "final": True
            }

        # Rule 4: Check if output matches prompt intent
        if self._matches_prompt_intent(bayko_output, original_prompt):
            return {
                "decision": "approve",
                "reason": "Output matches prompt and quality is acceptable",
                "final": True
            }
        else:
            return {
                "decision": "refine",
                "reason": "Output needs improvement to better match prompt",
                "final": False
            }

    def _has_dialogue_in_images(self, output: dict) -> bool:
        """Check if panels mention dialogue in the image"""
        panels = output.get("panels", [])

        dialogue_keywords = [
            "speech bubble", "dialogue", "talking", "saying",
            "text in image", "speech", "conversation"
        ]

        for panel in panels:
            description = panel.get("description", "").lower()
            if any(keyword in description for keyword in dialogue_keywords):
                print(f"❌ Found dialogue in image: {description}")
                return True

        return False

    def _is_story_coherent(self, output: dict) -> bool:
        """Basic check for story coherence"""
        panels = output.get("panels", [])

        if len(panels) < 2:
            return True  # Single panel is always coherent

        # Check 1: All panels should have descriptions
        descriptions = [p.get("description", "") for p in panels]
        if any(not desc.strip() for desc in descriptions):
            print("❌ Some panels missing descriptions")
            return False

        # Check 2: Panels shouldn't be identical (no progression)
        if len(set(descriptions)) == 1:
            print("❌ All panels are identical - no story progression")
            return False

        # Check 3: Look for obvious incoherence keywords
        incoherent_keywords = [
            "unrelated", "random", "doesn't make sense",
            "no connection", "contradictory"
        ]

        full_text = " ".join(descriptions).lower()
        if any(keyword in full_text for keyword in incoherent_keywords):
            print("❌ Story contains incoherent elements")
            return False

        return True

    def _matches_prompt_intent(self, output: dict, prompt: str) -> bool:
        """Check if output generally matches the original prompt"""
        panels = output.get("panels", [])

        if not panels:
            return False

        # Simple keyword matching
        prompt_words = set(prompt.lower().split())
        panel_text = " ".join([p.get("description", "") for p in panels]).lower()
        panel_words = set(panel_text.split())

        # At least 20% of prompt words should appear in panel descriptions
        overlap = len(prompt_words.intersection(panel_words))
        match_ratio = overlap / len(prompt_words) if prompt_words else 0

        print(f"📊 Prompt match ratio: {match_ratio:.2f}")
        return match_ratio >= 0.2

    def reset(self):
        """Reset for new session"""
        self.attempt_count = 0
```

### Integration with Brown

```python
# Update Brown's review_output method
from services.simple_evaluator import SimpleEvaluator

class AgentBrown:
    def __init__(self, max_iterations: int = 3):
        # ... existing code
        self.evaluator = SimpleEvaluator()

    def review_output(self, bayko_response: Dict[str, Any], original_request: StoryboardRequest):
        """Review Bayko's output using simple evaluation logic"""

        print(f"🤖 Brown reviewing Bayko's output...")

        # Use simple evaluator
        evaluation = self.evaluator.evaluate(
            bayko_response,
            original_request.prompt
        )

        # Log to memory
        self.memory.add_message(
            "assistant",
            f"Evaluation: {evaluation['decision']} - {evaluation['reason']}"
        )

        if evaluation["decision"] == "approve":
            print(f"✅ Brown approved: {evaluation['reason']}")
            return self._create_approval_message(bayko_response, evaluation)

        elif evaluation["decision"] == "reject":
            print(f"❌ Brown rejected: {evaluation['reason']}")
            return self._create_rejection_message(bayko_response, evaluation)

        else:  # refine
            print(f"🔄 Brown requesting refinement: {evaluation['reason']}")
            return self._create_refinement_message(bayko_response, evaluation)
```

---
