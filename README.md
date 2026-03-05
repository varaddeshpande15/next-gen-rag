# NXT_GEN-RAG — Document Intelligence Pipeline

> **Hackathon 2026** · A production-grade, offline-first RAG system that transforms handwritten & printed PDFs into a queryable knowledge base using Sarvam Vision OCR, semantic chunking, FAISS vector search, CrossEncoder reranking, and Qwen 2.5 LLM — all running locally on CPU/4 GB VRAM.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?logo=ollama)
![FAISS](https://img.shields.io/badge/FAISS-CPU-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Dual OCR** | Sarvam Vision API (cloud, 23 languages) with automatic PaddleOCR + OpenCV fallback for offline use |
| **Semantic Chunking** | LlamaIndex `SentenceSplitter` for context-safe chunks (300 tokens, 50 overlap) |
| **Nomic Embeddings** | `nomic-embed-text` via Ollama — 768-dim CPU-based embeddings |
| **FAISS Vector Store** | `IndexFlatIP` with cosine similarity, persistent index on disk |
| **Two-Stage Retrieval** | FAISS top-10 recall → CrossEncoder (`ms-marco-MiniLM-L-6-v2`) reranking to top-3 |
| **Hallucination Guards** | Hard cosine (`< 0.35`) and reranker (`< 0.35`) thresholds block LLM calls |
| **LLM Generation** | Qwen 2.5-3B via Ollama with strict grounding system prompt |
| **Confidence Scoring** | Secondary LLM self-evaluation (1–5 scale) with color-coded badges |
| **Conversation Memory** | 10-turn session history per document for follow-up questions |
| **Multilingual** | Auto-detects 11 languages (Hindi, Tamil, Telugu, Bengali, etc.) via `langdetect` |
| **PDF Export** | ReportLab-powered session reports with all Q&A pairs and citations |
| **Premium UI** | Glassmorphic dark-mode interface with real-time upload, chat, and context panels |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      Frontend (index.html)                     │
│   Upload Zone  │  Chat Interface  │  Context Panel  │  Export  │
└───────┬────────┴────────┬─────────┴────────┬────────┴──────┬──┘
        │ POST /api/upload│ POST /api/query  │ GET /export   │ DELETE
        ▼                 ▼                  ▼               ▼
┌────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (main.py)                    │
├────────────┬──────────────┬──────────────┬─────────────────────┤
│    OCR     │     RAG      │  Retrieval   │   Bonus Features    │
│            │              │              │                     │
│ Sarvam API │ Chunker      │ FAISS Search │ Confidence Scoring  │
│ PaddleOCR  │ Embedder     │ CrossEncoder │ Conversation Memory │
│ OpenCV     │ VectorStore  │ Generator    │ Multilingual        │
│            │              │ (Qwen 2.5)   │ PDF Report          │
└────────────┴──────────────┴──────────────┴─────────────────────┘
        │                         │
        ▼                         ▼
   Ollama Server            FAISS Index
  (nomic-embed-text)        (faiss_index.bin)
  (qwen2.5:3b)              (faiss_metadata.pkl)
```

---

## 📁 Project Structure

```
nxt_gen_RAG/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app with all API routes
│   │   ├── ocr/
│   │   │   ├── __init__.py         # OCR router (Sarvam → PaddleOCR fallback)
│   │   │   ├── sarvam_ocr.py       # Sarvam Vision Document Intelligence
│   │   │   ├── paddle_ocr.py       # Local PaddleOCR fallback
│   │   │   └── preprocess.py       # OpenCV deskew, denoise, binarize
│   │   ├── rag/
│   │   │   ├── chunker.py          # SentenceSplitter (300 tokens)
│   │   │   ├── embedder.py         # nomic-embed-text via Ollama
│   │   │   ├── vectorstore.py      # FAISS IndexFlatIP with persistence
│   │   │   ├── retriever.py        # Two-stage FAISS + CrossEncoder
│   │   │   └── generator.py        # Qwen 2.5-3B answer generation
│   │   ├── features/
│   │   │   ├── confidence.py       # LLM self-evaluation (1-5 score)
│   │   │   ├── memory.py           # 10-turn session memory
│   │   │   ├── multilingual.py     # langdetect + language mapping
│   │   │   └── report.py           # ReportLab PDF generation
│   │   └── utils/
│   │       └── prompts.py          # System prompts for LLM
│   ├── requirements.txt
│   └── venv/
├── frontend/
│   └── index.html                  # Full single-file UI (HTML + CSS + JS)
├── .env                            # SARVAM_API_KEY
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Ollama** installed and running ([install guide](https://ollama.ai))
- **Poppler** (for `pdf2image`): `sudo apt install poppler-utils`

### 1. Pull Ollama Models

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

### 2. Setup Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
SARVAM_API_KEY=your_sarvam_api_key_here
```

> **Note:** The Sarvam API key is optional. If unavailable or if the API fails, the system automatically falls back to local PaddleOCR.

### 4. Start the Server

```bash
cd backend/app
../venv/bin/uvicorn main:app --reload --port 8000
```

### 5. Open the Frontend

Open `frontend/index.html` in your browser. The UI will connect to `http://localhost:8000`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/upload` | Upload PDF → OCR → Chunk → Embed → Index |
| `GET` | `/api/upload/{file_id}/status` | Check upload/indexing status |
| `POST` | `/api/query` | Ask a question against indexed documents |
| `GET` | `/api/export-report?file_id=` | Download PDF report of session Q&A |
| `DELETE` | `/api/session` | Clear all documents and memory |

### Query Request Body

```json
{
  "question": "Tell me about activity based scheduling",
  "file_ids": ["962b8552"],
  "language": "en",
  "memory": []
}
```

### Query Response

```json
{
  "answer": "Activity Based Scheduling is a project scheduling approach...",
  "confidence": "high",
  "conf_score": 5,
  "conf_reason": "Context directly explains the topic",
  "citations": [
    { "page": "notes.pdf · Page 3", "section": "Notes", "text": "..." }
  ],
  "chunks": [
    { "source": "notes.pdf · Page 3", "score": "6.71", "text": "..." }
  ]
}
```

---

## 🛡️ Hallucination Prevention

The system employs a **three-layer defense** against hallucination:

1. **Cosine Similarity Guard** — If the best FAISS match scores below `0.35`, no LLM call is made.
2. **CrossEncoder Reranker Guard** — If the best reranked score is below `0.35`, the system refuses to answer.
3. **Strict System Prompt** — The LLM is instructed to respond only from provided context and say _"I don't have enough information"_ otherwise.
4. **Confidence Auto-Override** — If the self-evaluation scores ≤ 2/5, the answer is overridden with a safe refusal.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI + Uvicorn |
| **OCR (Primary)** | Sarvam Vision Document Intelligence API |
| **OCR (Fallback)** | PaddleOCR + OpenCV preprocessing |
| **Chunking** | LlamaIndex SentenceSplitter |
| **Embeddings** | nomic-embed-text (768-dim, via Ollama) |
| **Vector Store** | FAISS CPU (IndexFlatIP, cosine similarity) |
| **Reranker** | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| **LLM** | Qwen 2.5-3B (via Ollama, ~2 GB VRAM) |
| **Confidence** | Secondary LLM self-evaluation call |
| **Language Detection** | langdetect (11 Indian languages + English) |
| **PDF Export** | ReportLab |
| **Frontend** | Vanilla HTML/CSS/JS (single file, glassmorphic UI) |

---

## 📦 Dependencies

```
fastapi
uvicorn
python-multipart
python-dotenv
requests
sarvamai
llama-index
llama-index-embeddings-ollama
faiss-cpu
sentence-transformers
reportlab
langdetect
opencv-python
paddlepaddle
paddleocr
pdf2image
```

---

## 🌐 Multilingual Support

Auto-detects and supports queries in:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | `en` | Bengali | `bn` |
| Hindi | `hi` | Marathi | `mr` |
| Tamil | `ta` | Gujarati | `gu` |
| Telugu | `te` | Kannada | `kn` |
| Punjabi | `pa` | Malayalam | `ml` |
| Urdu | `ur` | | |

---

## 📄 License

MIT License — built for Gen AI Hackathon 2026.
