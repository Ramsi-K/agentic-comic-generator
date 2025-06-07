from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage


class AgentMemory:
    """Simple wrapper around LlamaIndex Memory for agent conversations"""

    def __init__(self, session_id: str, agent_name: str):
        self.session_id = session_id
        self.agent_name = agent_name

        # Use LlamaIndex Memory with session-specific ID
        self.memory = Memory.from_defaults(
            session_id=f"{session_id}_{agent_name}", token_limit=4000
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
