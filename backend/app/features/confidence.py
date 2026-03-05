import requests
from utils.prompts import CONFIDENCE_PROMPT
import logging

logger = logging.getLogger(__name__)

def evaluate_confidence(query: str, answer: str, context: str) -> dict:
    prompt = CONFIDENCE_PROMPT.format(context=context, question=query, answer=answer)
    
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen2.5:3b",
        "messages": [{'role': 'user', 'content': prompt}],
        "stream": False
    })
    
    score = 3
    reason = "Unable to properly score."
    badge = "medium"
    
    if resp.status_code == 200:
        output = resp.json()['message']['content']
        try:
            for line in output.split('\n'):
                if line.startswith("Score:"):
                    score = int(line.replace("Score:", "").strip())
                elif line.startswith("Reason:"):
                    reason = line.replace("Reason:", "").strip()
        except:
            pass
            
    if score >= 4:
        badge = "high"
    elif score <= 2:
        badge = "low"
        
    return {"score": score, "reason": reason, "badge": badge}
