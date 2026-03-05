import logging
from typing import List
import requests

logger = logging.getLogger(__name__)

def embed_chunks(chunks: List[dict]) -> List[List[float]]:
    logger.info(f"Embedding {len(chunks)} chunks using nomic-embed-text...")
    embeddings = []
    for c in chunks:
        resp = requests.post("http://localhost:11434/api/embed", json={
            "model": "nomic-embed-text",
            "input": c["text"]
        })
        if resp.status_code == 200:
            data = resp.json()
            if 'embeddings' in data and len(data['embeddings']) > 0:
                embeddings.append(data['embeddings'][0])
            else:
                logger.warning("No embeddings returned.")
        else:
            logger.error(f"Embedding failed: {resp.text}")
    return embeddings
