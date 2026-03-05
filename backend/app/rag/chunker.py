import logging
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from typing import List, Dict

logger = logging.getLogger(__name__)

# Switch to SentenceSplitter to prevent context length errors (the 400 error)
# when OCR returns massive blobs without punctuation.
splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)

def chunk_pages_semantically(pages: List[Dict], file_id: str, filename: str) -> List[Dict]:
    logger.info(f"Chunking {len(pages)} pages using SentenceSplitter...")
    docs = [Document(text=p["text"], metadata={"page_num": p["page_num"], "file_id": file_id, "filename": filename}) for p in pages]
    nodes = splitter.get_nodes_from_documents(docs)
    
    chunks = []
    for node in nodes:
        chunks.append({
            "text": node.text,
            "metadata": node.metadata
        })
    return chunks
