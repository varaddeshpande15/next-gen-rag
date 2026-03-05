from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import uuid
import os
import tempfile
import logging

from ocr import process_pdf
from rag.chunker import chunk_pages_semantically
from rag.embedder import embed_chunks
from rag.vectorstore import VectorStore
from rag.retriever import retrieve_and_rerank
from rag.generator import generate_answer
from features.confidence import evaluate_confidence
from features.memory import get_memory, add_memory, clear_memory
from features.multilingual import detect_language, get_language_name
from features.report import generate_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="nxt_gen_RAG API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

document_store = {}
vector_store = VectorStore(index_file="/home/varad-deshpande/dev/nxt_gen_RAG/backend/faiss_index.bin", meta_file="/home/varad-deshpande/dev/nxt_gen_RAG/backend/faiss_metadata.pkl")

class MemoryItem(BaseModel):
    role: str
    text: str

class QueryRequest(BaseModel):
    question: str
    file_ids: list[str] = []
    language: str = "auto"
    memory: list[MemoryItem] = []

class Citation(BaseModel):
    page: str
    section: str
    text: str

class Chunk(BaseModel):
    source: str
    score: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    confidence: str
    conf_score: int
    conf_reason: str
    citations: list[Citation]
    chunks: list[Chunk]

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    pages: int
    status: str
    chunks: int

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_id = str(uuid.uuid4())[:8]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        bytes_data = await file.read()
        tmp.write(bytes_data)
        tmp_path = tmp.name

    try:
        pages = await process_pdf(tmp_path)
        if not pages:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF")

        chunks = chunk_pages_semantically(pages, file_id, file.filename)
        embeddings = embed_chunks(chunks)
        vector_store.add_chunks(chunks, embeddings)
        
        document_store[file_id] = {"filename": file.filename, "pages": len(pages), "chunks": len(chunks)}
        clear_memory(file_id)
        
        return UploadResponse(file_id=file_id, filename=file.filename, pages=len(pages), status="indexed", chunks=len(chunks))
    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/api/upload/{file_id}/status")
async def upload_status(file_id: str):
    if file_id not in document_store:
        raise HTTPException(status_code=404, detail="File not found")
    return {"file_id": file_id, "status": "indexed", "progress": 100, "pages": document_store[file_id]["pages"], "chunks": document_store[file_id]["chunks"]}

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
        
    session_id = request.file_ids[0] if request.file_ids else "global_session"
    lang_code = detect_language(request.question)
    lang_name = get_language_name(lang_code)
    
    top_chunks, max_rerank = retrieve_and_rerank(request.question, vector_store)
    
    if not top_chunks:
        ans = "I don't have enough information in the notes to answer this."
        return _build_response(ans, conf=1, reason="No relevant context retrieved or failed relevance threshold.", badge="low")

    history = get_memory(session_id)
    gen_result = generate_answer(request.question, top_chunks, history)
    answer_text = gen_result["answer"]
    
    if "don't have enough information" in answer_text.lower():
        return _build_response(answer_text, conf=1, reason="LLM detected insufficient information.", badge="low", chunks=top_chunks, citations=gen_result["citations"])

    context_str = " ".join([c["text"] for c in top_chunks])
    eval_result = evaluate_confidence(request.question, answer_text, context_str)
    
    if eval_result["score"] <= 2:
        answer_text = "I don't have enough information in the notes to answer this securely."
        eval_result["reason"] = "Auto-override triggered due to low confidence."
        
    if lang_code != 'en':
        answer_text = f"{answer_text}\n\n*Translated internally based on query language ({lang_name}).*"

    add_memory(session_id, "user", request.question)
    add_memory(session_id, "assistant", answer_text)

    return _build_response(answer_text, eval_result["score"], eval_result["reason"], eval_result["badge"], chunks=top_chunks, citations=gen_result["citations"])
    
def _build_response(answer: str, conf: int, reason: str, badge: str, chunks=None, citations=None) -> QueryResponse:
    if chunks is None: chunks = []
    if citations is None: citations = []
    formatted_chunks = [Chunk(source=f"{c['metadata'].get('filename', 'Unknown')} · Page {c['metadata'].get('page_num', '?')}", score=str(c.get("rerank_score", "Rerank Score")), text=c["text"][:200] + "...") for c in chunks]
    return QueryResponse(answer=answer, confidence=badge, conf_score=conf, conf_reason=reason, citations=citations, chunks=formatted_chunks)

@app.get("/api/export-report")
async def export_report(file_id: str = "global_session"):
    try:
        if file_id == "global_session" and document_store:
            file_id = list(document_store.keys())[0]
            
        history = get_memory(file_id)
        title = document_store.get(file_id, {}).get("filename", "Session")
        report_path = generate_report(history, filename_context=title)
        return FileResponse(report_path, media_type="application/pdf", filename=f"Report_{title}.pdf")
    except Exception as e:
        logger.error(f"Report export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session")
async def clear_session():
    document_store.clear()
    vector_store.clear()
    return {"status": "cleared", "message": "All documents removed from memory"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)