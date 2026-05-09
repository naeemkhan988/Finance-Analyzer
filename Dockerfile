FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves ~1.5GB vs full torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model during build
# so it doesn't need to download at runtime (prevents startup timeout)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Create required data directories
RUN mkdir -p data/raw data/processed data/embeddings logs

# Railway provides PORT env var — default to 8000 for local testing
ENV PORT=8000

# Start the unified server (FastAPI + Flask frontend on same port)
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 120 --workers 1
