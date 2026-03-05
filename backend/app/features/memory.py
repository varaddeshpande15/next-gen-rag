from typing import List, Dict

session_memory = {}
MAX_TURNS = 10

def get_memory(session_id: str) -> List[Dict]:
    return session_memory.get(session_id, [])

def add_memory(session_id: str, role: str, text: str):
    if session_id not in session_memory:
        session_memory[session_id] = []
    session_memory[session_id].append({"role": role, "text": text})
    if len(session_memory[session_id]) > MAX_TURNS:
        session_memory[session_id] = session_memory[session_id][-MAX_TURNS:]

def clear_memory(session_id: str):
    if session_id in session_memory:
        del session_memory[session_id]
