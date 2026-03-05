from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder
import requests
import logging

logger = logging.getLogger(__name__)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_query_embedding(query: str) -> List[float]:
    resp = requests.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-text",
        "input": query
    })
    if resp.status_code == 200:
        return resp.json()['embeddings'][0]
    return []

def retrieve_and_rerank(query: str, vector_store) -> Tuple[List[Dict], float]:
    q_emb = get_query_embedding(query)
    if not q_emb:
        return [], 0.0
        
    faiss_results = vector_store.search(q_emb, top_k=10)
    if not faiss_results:
        return [], 0.0
        
    # Check max cosine
    max_cosine = faiss_results[0][0]
    if max_cosine < 0.35:
        logger.warning(f"Cosine too low: {max_cosine}")
        return [], 0.0

    pairs = [[query, chunk["text"]] for score, chunk in faiss_results]
    rerank_scores = reranker.predict(pairs)
    
    reranked_results = list(zip(rerank_scores, [chunk for score, chunk in faiss_results]))
    reranked_results.sort(key=lambda x: x[0], reverse=True)
    
    max_rerank = reranked_results[0][0]
    if max_rerank < 0.35:
        logger.warning(f"Rerank score too low: {max_rerank}")
        return [], 0.0
        
    top_3 = [chunk for score, chunk in reranked_results][:3]
    for i, chunk in enumerate(top_3):
        chunk["rerank_score"] = float(reranked_results[i][0])
        
    return top_3, max_rerank
