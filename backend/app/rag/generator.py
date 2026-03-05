import logging
from typing import List, Dict
import requests
from utils.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def generate_answer(query: str, chunks: List[Dict], memory: List[Dict] = None) -> Dict:
    context_text = "\n\n".join([f"[Source: {c['metadata'].get('filename', 'Unknown')} - Page {c['metadata'].get('page_num', '?')}]\n{c['text']}" for c in chunks])
    
    memory = memory or []
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in memory])
    
    user_prompt = f"Context:\n{context_text}\n\nHistory:\n{history_text}\n\nQuestion:\n{query}"
    
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen2.5:3b",
        "messages": [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ],
        "stream": False
    })
    
    answer = ""
    if resp.status_code == 200:
        answer = resp.json()['message']['content']
        
    citations = []
    for c in chunks:
        citations.append({
            "page": f"{c['metadata'].get('filename', 'Unknown')} · Page {c['metadata'].get('page_num', '?')}",
            "section": "Notes",
            "text": c["text"][:100] + "..."
        })
        
    return {
        "answer": answer,
        "citations": citations
    }
