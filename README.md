# Multimodal RAG Financial Document Analyzer

Production-grade Retrieval-Augmented Generation (RAG) system for intelligent financial document analysis. Upload PDFs, DOCX, XLSX, or CSV files and query them with natural language — powered by hybrid vector search, multimodal fusion, and multi-provider LLM generation.

---

## Architecture

```
┌─────────────────┐       ┌──────────────────────────────────────────────────────┐
│                 │       │                    Backend (FastAPI :8000)            │
│  User Browser   │       │                                                      │
│                 │  HTTP │  ┌──────────┐   ┌──────────────────────────────────┐ │
│  ┌───────────┐  │◄─────►│  │  Routes   │   │        RAG Pipeline              │ │
│  │   Flask    │  │       │  │ /api/*    │──►│                                  │ │
│  │ Frontend   │  │       │  └──────────┘   │  ┌────────┐   ┌──────────────┐  │ │
│  │  (:5000)   │  │       │                  │  │Document│   │   Hybrid      │  │ │
│  └───────────┘  │       │  ┌──────────┐   │  │Loader  │──►│  Retriever    │  │ │
│                 │       │  │   Auth    │   │  └────────┘   │(Vector+BM25) │  │ │
│                 │       │  │Middleware │   │                └──────┬───────┘  │ │
└─────────────────┘       │  └──────────┘   │                       │          │ │
                          │                  │  ┌────────┐   ┌──────▼───────┐  │ │
                          │  ┌──────────┐   │  │Semantic│   │    FAISS     │  │ │
                          │  │   Rate   │   │  │ Cache  │   │ Vector Store │  │ │
                          │  │ Limiter  │   │  └────────┘   └──────────────┘  │ │
                          │  └──────────┘   │                                  │ │
                          │                  │  ┌────────┐   ┌──────────────┐  │ │
                          │                  │  │Reranker│──►│  LLM Router  │  │ │
                          │                  │  └────────┘   │(Groq/Gemini) │  │ │
                          │                  │               └──────────────┘  │ │
                          │                  └──────────────────────────────────┘ │
                          │                                                      │
                          │  ┌─────────────────────────────────────────────────┐ │
                          │  │              SQLite Storage                      │ │
                          │  │  conversations │ documents │ api_keys │ cache   │ │
                          │  └─────────────────────────────────────────────────┘ │
                          └──────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/finance-analyzer.git
cd finance-analyzer
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (at minimum one of GROQ_API_KEY or GOOGLE_API_KEY)
```

### 3. Start Backend (FastAPI)

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start Frontend (Flask) — in a separate terminal

```bash
python frontend/app.py
```

### 5. Open Browser

Navigate to `http://localhost:5000`

---

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `SECRET_KEY` | Flask session encryption key | `my-super-secret-key-change-me` |
| `GROQ_API_KEY` | Groq API key for Llama 3.3 70B (free tier) | `gsk_abc123...` |
| `GROQ_MODEL` | Groq model name | `llama-3.3-70b-versatile` |
| `GOOGLE_API_KEY` | Google Gemini API key (free tier) | `AIza...` |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | Sentence-transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `384` |
| `FAISS_INDEX_PATH` | Directory for FAISS index files | `./data/embeddings` |
| `FAISS_INDEX_NAME` | FAISS collection/index name | `financial_docs` |
| `CHUNK_SIZE` | Document chunk size in characters | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks | `128` |
| `MAX_FILE_SIZE_MB` | Maximum upload file size in MB | `500` |
| `UPLOAD_FOLDER` | Directory for uploaded documents | `./data/raw` |
| `DATABASE_URL` | SQLite database path | `sqlite:///./data/app.db` |
| `ADMIN_API_KEY` | Admin API key (auto-generates if missing) | `your-admin-key` |
| `CACHE_SIMILARITY_THRESHOLD` | Semantic cache cosine similarity threshold | `0.92` |
| `CACHE_TTL_HOURS` | Cache entry time-to-live in hours | `24` |
| `RATE_LIMIT_FREE_PER_HOUR` | Rate limit for free tier (requests/hour) | `20` |
| `RATE_LIMIT_PRO_PER_HOUR` | Rate limit for pro tier (requests/hour) | `200` |
| `STREAM_ENABLED` | Enable SSE streaming responses | `true` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FILE` | Log file path | `./logs/app.log` |

---

## API Endpoints

### Health & Status

```
GET /api/health
→ {"status": "healthy", "timestamp": "2026-05-01T12:00:00"}

GET /api/stats
→ {"status": "operational", "pipeline_active": true, "documents": 3, ...}

GET /api/providers
→ {"groq": {"name": "Groq (Llama 3.3 70B)", "active": true, ...}}

GET /api/analytics?days=7
→ {"query_volume": [...], "top_questions": [...], "cache_stats": {...}}

GET /api/evaluate?suite=financial_qa
→ {"success": true, "metrics": {"avg_precision": 0.95, ...}, "summary": {...}}
```

### Document Management

```
POST /api/upload
Content-Type: multipart/form-data
Body: file=@report.pdf
→ {"success": true, "filename": "report.pdf", "chunks_created": 42, ...}

GET /api/documents
→ {"documents": [{"source": "report.pdf", "chunks": 42, ...}], "total": 1}

DELETE /api/documents/{filename}
→ {"success": true, "message": "Document 'report.pdf' deleted"}

GET /api/documents/{filename}/preview
→ {"filename": "report.pdf", "summary": {...}, "preview_pages": [...]}
```

### Query & Chat

```
POST /api/query
Content-Type: application/json
Body: {"question": "What was total revenue?", "session_id": "abc", "k": 5}
→ {"success": true, "answer": "Total revenue was $10.5 billion...", "sources": [...]}

POST /api/query/stream
Content-Type: application/json
Body: {"question": "Compare Q3 vs Q4", "session_id": "abc"}
→ SSE stream: {"type": "token", "content": "..."} ...

GET /api/history/{session_id}
→ {"session_id": "abc", "history": [{"question": "...", "answer": "...", ...}]}

DELETE /api/history/{session_id}
→ {"success": true, "message": "Session 'abc' cleared"}
```

---

## Example Queries

| Query | Expected Output |
|---|---|
| "What was the total revenue?" | Extracts revenue figures with page citations |
| "Compare Q3 vs Q4 profits" | Side-by-side comparison from financial tables |
| "Key risk factors mentioned?" | Lists risk factors with source documents |
| "Gross margin trends over the last year" | Trend analysis with percentage data |
| "Summarize the balance sheet" | Structured summary of assets and liabilities |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_integration.py -v
pytest tests/test_security.py -v

# Run with coverage
pytest tests/ --cov=src --cov=backend --cov-report=term-missing
```

---

## Docker Deployment

### Using Docker Compose

```bash
# Build and start all services
docker-compose -f docker/docker-compose.yml up --build -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

### Services

| Service | Port | Description |
|---|---|---|
| `nginx` | 80/443 | Reverse proxy, SSL termination, static caching |
| `backend` | 8000 | FastAPI backend with RAG pipeline |
| `frontend` | 5000 | Flask frontend with premium UI |

### Production Checklist

1. Set strong `SECRET_KEY` in `.env`
2. Configure at least one LLM provider API key
3. Set `ADMIN_API_KEY` or let it auto-generate on first run
4. Mount persistent volume for `data/` directory
5. Configure SSL certificates in nginx

---

## Project Structure

```
finance-analyzer/
├── backend/              # FastAPI backend
│   ├── core/             # Config & dependencies
│   ├── routes/           # API endpoints (upload, query, health)
│   ├── middleware/        # Auth & rate limiting middleware
│   ├── schemas/          # Pydantic request/response models
│   └── main.py           # Application entry point
├── frontend/             # Flask frontend
│   ├── static/           # CSS, JS assets
│   ├── templates/        # Jinja2 HTML templates
│   ├── utils/            # API client
│   └── app.py            # Flask application
├── src/                  # Core business logic
│   ├── auth/             # API key management & rate limiting
│   ├── cache/            # Semantic caching
│   ├── database/         # SQLite conversation & document storage
│   ├── models/           # LLM router & embedding models
│   ├── multimodal/       # Image, table, OCR extraction
│   ├── rag/              # Pipeline, retriever, vector store
│   └── utils/            # Logging, file utilities
├── docker/               # Docker configuration
├── tests/                # Test suite
├── data/                 # Runtime data (uploads, embeddings, DB)
└── config/               # YAML configuration files
```

---

## Troubleshooting (Windows)

1. **ModuleNotFoundError**: Ensure you activated the virtual environment: `.\venv\Scripts\activate`.
2. **Tesseract Not Found**: Install Tesseract OCR from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add `C:\Program Files\Tesseract-OCR` to your System PATH.
3. **401 Unauthorized**: Ensure `ADMIN_API_KEY` is set in your `.env` file. The frontend uses this to authenticate with the backend.
4. **SQLite Errors**: Ensure the `data/` directory is writable.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
