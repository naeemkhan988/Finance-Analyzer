"""
Unified Flask Application
==========================
Single Flask app serving both UI and API for the
Multimodal RAG Financial Document Analyzer.

Run:   python app.py
Deploy: gunicorn app:app --bind 0.0.0.0:$PORT
"""

import os
import sys
import uuid
import logging
import time
from pathlib import Path
from datetime import datetime

# ──── Paths ────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for,
)
from flask_cors import CORS

from src.utils.logger import setup_logging

# ──── Logging ──────────────────────────────────────────────
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ──── Upload Config ────────────────────────────────────────
UPLOAD_DIR = Path(os.getenv("UPLOAD_FOLDER", "./data/raw"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "xlsx", "csv"}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024

# ──── RAG Pipeline (singleton) ─────────────────────────────
_rag_pipeline = None


def get_rag_pipeline():
    """Get or create the RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        try:
            from src.rag.pipeline import RAGPipeline

            _rag_pipeline = RAGPipeline(
                chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 128)),
                embedding_model=os.getenv(
                    "EMBEDDING_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ),
                embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", 384)),
                index_path=os.getenv("FAISS_INDEX_PATH", "./data/embeddings"),
                collection_name=os.getenv("FAISS_INDEX_NAME", "financial_docs"),
            )
            logger.info("✅ RAG Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"❌ RAG Pipeline init failed: {e}")
            raise
    return _rag_pipeline


# ──── Create Flask App ─────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
CORS(app)


# ──── Initialize pipeline at startup ──────────────────────
with app.app_context():
    try:
        pipeline = get_rag_pipeline()
        logger.info(
            f"🚀 Finance Analyzer ready  |  "
            f"Env={os.getenv('FLASK_ENV', 'development')}"
        )
    except Exception as e:
        logger.warning(f"Pipeline not ready yet: {e}")


# ═══════════════════════════════════════════════════════════
#   TEMPLATE CONTEXT
# ═══════════════════════════════════════════════════════════

@app.context_processor
def inject_globals():
    """Inject global template variables."""
    return {
        "app_name": "Financial Document Analyzer",
        "app_version": "1.0.0",
    }


# ═══════════════════════════════════════════════════════════
#   PAGE ROUTES  (serve HTML)
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Landing / Chat page."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())[:8]
    if "messages" not in session:
        session["messages"] = []
    return render_template("index.html", messages=session.get("messages", []))


@app.route("/upload")
def upload_page():
    """Upload documents page."""
    documents = _list_documents_internal()
    return render_template("upload.html", documents=documents)


@app.route("/dashboard")
def dashboard_page():
    """System dashboard page."""
    stats = _get_stats_internal()
    providers = _get_providers_internal()
    return render_template("dashboard.html", stats=stats, providers=providers)


@app.route("/history")
def history_page():
    """Conversation history page."""
    session_id = session.get("session_id", "default")
    history = _get_history_internal(session_id)
    return render_template("history.html", history=history, session_id=session_id)


# ═══════════════════════════════════════════════════════════
#   API ROUTES  (JSON endpoints — called by JS in browser)
# ═══════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    pipeline = _pipeline_or_none()
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "multimodal-rag-finance-analyzer",
        "pipeline_active": pipeline is not None,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload and ingest a financial document."""
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file selected"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected"}), 400

    # Validate extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "message": f"File type '.{ext}' not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        }), 400

    # Save file
    file_path = UPLOAD_DIR / file.filename
    try:
        contents = file.read()
        if len(contents) > MAX_FILE_SIZE:
            return jsonify({
                "success": False,
                "message": f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB",
            }), 400

        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"File saved: {file.filename} ({len(contents)} bytes)")
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to save file: {e}"}), 500

    # Ingest into RAG pipeline
    pipeline = _pipeline_or_none()
    if pipeline:
        try:
            result = pipeline.ingest_document(str(file_path))
            return jsonify({
                "success": result["success"],
                "filename": file.filename,
                "message": f"Document ingested: {result.get('chunks_created', 0)} chunks created",
                "chunks_created": result.get("chunks_created", 0),
                "processing_time": result.get("processing_time_seconds", 0),
            })
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return jsonify({
                "success": False,
                "filename": file.filename,
                "message": f"File saved but ingestion failed: {e}",
            })
    else:
        return jsonify({
            "success": True,
            "filename": file.filename,
            "message": "File saved. RAG pipeline not initialized — please check logs.",
        })


@app.route("/api/query", methods=["POST"])
def api_query():
    """Query documents using the RAG pipeline."""
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"success": False, "answer": "Please enter a question."}), 400

    pipeline = _pipeline_or_none()
    if not pipeline:
        return jsonify({
            "success": False,
            "answer": "RAG pipeline not initialized. Please upload a document first.",
        }), 503

    session_id = session.get("session_id", "default")
    try:
        result = pipeline.query(
            question=question,
            session_id=session_id,
            k=data.get("k", 5),
            filter_source=data.get("filter_source"),
            use_hybrid=data.get("use_hybrid", True),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return jsonify({"success": False, "answer": f"Error: {str(e)}"})

    # Save messages to session
    messages = session.get("messages", [])
    messages.append({"role": "user", "content": question})
    messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {}),
    })
    session["messages"] = messages[-40:]

    return jsonify(result)


@app.route("/api/documents", methods=["GET"])
def api_documents():
    """List uploaded documents."""
    docs = _list_documents_internal()
    return jsonify({"documents": docs, "total": len(docs)})


@app.route("/api/documents/<filename>", methods=["DELETE"])
def api_delete_document(filename):
    """Delete a document."""
    pipeline = _pipeline_or_none()
    if pipeline:
        result = pipeline.delete_document(filename)
    else:
        result = {"success": True}

    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        os.remove(file_path)

    return jsonify(result)


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """System statistics."""
    return jsonify(_get_stats_internal())


@app.route("/api/providers", methods=["GET"])
def api_providers():
    """LLM provider information."""
    return jsonify(_get_providers_internal())


@app.route("/api/clear-chat", methods=["POST"])
def api_clear_chat():
    """Clear chat history."""
    session["messages"] = []
    session_id = session.get("session_id", "default")
    pipeline = _pipeline_or_none()
    if pipeline:
        pipeline.clear_conversation(session_id)
    return jsonify({"success": True})


@app.route("/api/history/<session_id>", methods=["GET"])
def api_history(session_id):
    """Get conversation history for a session."""
    history = _get_history_internal(session_id)
    return jsonify({"session_id": session_id, "history": history})


@app.route("/api/history/<session_id>", methods=["DELETE"])
def api_clear_history(session_id):
    """Clear conversation history for a session."""
    pipeline = _pipeline_or_none()
    if pipeline:
        pipeline.clear_conversation(session_id)
    return jsonify({"success": True, "message": f"Session '{session_id}' cleared"})


# ═══════════════════════════════════════════════════════════
#   INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════

def _pipeline_or_none():
    """Return the pipeline if initialized, else None."""
    global _rag_pipeline
    return _rag_pipeline


def _list_documents_internal():
    """List documents from pipeline or filesystem."""
    pipeline = _pipeline_or_none()
    if pipeline:
        try:
            return pipeline.get_documents()
        except Exception:
            pass

    # Fallback: list files from upload directory
    files = []
    for path in UPLOAD_DIR.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            files.append({
                "filename": path.name,
                "source": path.name,
                "size_bytes": path.stat().st_size,
            })
    return files


def _get_stats_internal():
    """Get system statistics."""
    pipeline = _pipeline_or_none()
    if pipeline:
        try:
            stats = pipeline.get_stats()
            return {"status": "operational", "pipeline_active": True, **stats}
        except Exception:
            pass
    return {"status": "limited", "pipeline_active": False, "message": "RAG pipeline not initialized"}


def _get_providers_internal():
    """Get LLM provider info."""
    pipeline = _pipeline_or_none()
    if pipeline and hasattr(pipeline, "llm_router"):
        try:
            return pipeline.llm_router.get_provider_info()
        except Exception:
            pass
    return {"fallback": {"name": "No providers configured", "active": True}}


def _get_history_internal(session_id):
    """Get conversation history."""
    pipeline = _pipeline_or_none()
    if pipeline:
        try:
            return pipeline.get_conversation_history(session_id)
        except Exception:
            pass
    return []


# ═══════════════════════════════════════════════════════════
#   ERROR HANDLERS
# ═══════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return render_template("base.html", error="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("base.html", error="Internal server error"), 500


# ═══════════════════════════════════════════════════════════
#   RUN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("APP_PORT", 5000)))
    app.run(
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=port,
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
    )
