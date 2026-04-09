"""
Flask Frontend Application
===========================
Premium UI for the Multimodal RAG Financial Document Analyzer.
"""

import os
import sys
import uuid
import logging
from pathlib import Path

# Add project root AND frontend dir to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
FRONTEND_DIR = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS

# Import using direct path (avoids reloader issues)
from utils.api_client import APIClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
CORS(app)

# API Client
api_client = APIClient()


# ─── Context Processors ────────────────────────────────────────
@app.context_processor
def inject_globals():
    """Inject global template variables."""
    return {
        "app_name": "Financial Document Analyzer",
        "app_version": "1.0.0",
    }


# ─── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    """Landing / Chat page."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())[:8]
    if "messages" not in session:
        session["messages"] = []

    return render_template("index.html", messages=session.get("messages", []))


@app.route("/upload", methods=["GET"])
def upload_page():
    """Upload documents page."""
    documents = api_client.list_documents()
    return render_template("upload.html", documents=documents)


@app.route("/dashboard")
def dashboard_page():
    """System dashboard page."""
    stats = api_client.get_stats()
    providers = api_client.get_providers()
    return render_template("dashboard.html", stats=stats, providers=providers)


@app.route("/history")
def history_page():
    """Conversation history page."""
    session_id = session.get("session_id", "default")
    history = api_client.get_history(session_id)
    return render_template("history.html", history=history, session_id=session_id)


# ─── API Proxy Routes ──────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def handle_upload():
    """Handle file upload and forward to backend API."""
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file selected"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected"}), 400

    result = api_client.upload_document(file)
    return jsonify(result)


@app.route("/api/query", methods=["POST"])
def handle_query():
    """Handle query and forward to backend API."""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"success": False, "answer": "Please enter a question."}), 400

    session_id = session.get("session_id", "default")

    result = api_client.query(
        question=question,
        session_id=session_id,
        k=data.get("k", 5),
        use_hybrid=data.get("use_hybrid", True),
    )

    # Save to session messages
    messages = session.get("messages", [])
    messages.append({"role": "user", "content": question})
    messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {}),
    })
    # Keep last 40 messages
    session["messages"] = messages[-40:]

    return jsonify(result)


@app.route("/api/documents", methods=["GET"])
def get_documents():
    """Get list of documents."""
    docs = api_client.list_documents()
    return jsonify({"documents": docs})


@app.route("/api/documents/<filename>", methods=["DELETE"])
def delete_document(filename):
    """Delete a document."""
    result = api_client.delete_document(filename)
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check proxy."""
    return jsonify(api_client.health_check())


@app.route("/api/stats", methods=["GET"])
def stats():
    """Stats proxy."""
    return jsonify(api_client.get_stats())


@app.route("/api/clear-chat", methods=["POST"])
def clear_chat():
    """Clear chat history."""
    session["messages"] = []
    session_id = session.get("session_id", "default")
    api_client.clear_history(session_id)
    return jsonify({"success": True})


# ─── Error Handlers ────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("base.html", error="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("base.html", error="Internal server error"), 500


# ─── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host=os.getenv("FRONTEND_HOST", "0.0.0.0"),
        port=int(os.getenv("FRONTEND_PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
    )
