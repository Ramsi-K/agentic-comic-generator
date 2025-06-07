from services.sqlite_memory import AgentMemory as SQLiteMemory
from services.llama_memory import AgentMemory as LlamaMemory


def sync_to_sqlite(llama_memory: LlamaMemory, sqlite_memory: SQLiteMemory):
    """Sync LlamaIndex memory to SQLite for persistence"""
    history = llama_memory.get_history()

    for message in history:
        sqlite_memory.add_message(
            session_id=llama_memory.session_id,
            agent_name=llama_memory.agent_name,
            content=message.content,
            step_type="message",
        )
