# Query routes — API endpoints for document querying, chat, and SSE streaming.
"""
Query Routes
============
API endpoints for document querying, chat, and SSE streaming responses.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Query request body."""
    question: str
    session_id: str = "default"
    k: int = 5
    filter_source: Optional[str] = None
    use_hybrid: bool = True


class QueryResponse(BaseModel):
    """Query response body."""
    success: bool
    answer: str = ""
    sources: list = []
    metadata: dict = {}


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, body: QueryRequest):
    """Query documents using the RAG pipeline."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please upload documents first.",
        )

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = pipeline.query(
            question=body.question,
            session_id=body.session_id,
            k=body.k,
            filter_source=body.filter_source,
            use_hybrid=body.use_hybrid,
        )

        return QueryResponse(
            success=True,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return QueryResponse(
            success=False,
            answer=f"Error processing query: {str(e)}",
        )


@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest):
    """
    Stream query results using Server-Sent Events (SSE).

    Event types:
    - status: Progress updates
    - cache_hit: Semantic cache hit notification
    - token: Individual response tokens
    - sources: Source citations
    - done: Stream complete
    - error: Error occurred
    """
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized.",
        )

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    def sse_event(event_type: str, data: dict) -> str:
        """Format a Server-Sent Event."""
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"

    async def event_generator():
        try:
            # Status: searching
            yield sse_event("status", {"message": "Searching documents..."})

            # Generate query embedding
            query_embedding = pipeline.embedding_manager.embed_query(body.question)

            # Check semantic cache
            cached = pipeline.semantic_cache.get(body.question, query_embedding)
            if cached is not None:
                similarity = cached.get("metadata", {}).get("cache_similarity", 0)
                yield sse_event("cache_hit", {"similarity": similarity})
                yield sse_event("token", {"content": cached["answer"]})
                yield sse_event("sources", {"sources": cached.get("sources", [])})

                # Save to conversation
                pipeline._update_conversation(
                    body.session_id, body.question, cached["answer"],
                    cached.get("sources", []), cached.get("metadata", {}),
                )
                yield sse_event("done", {})
                return

            # Retrieve
            filter_metadata = None
            if body.filter_source:
                filter_metadata = {"source": body.filter_source}

            retrieved = pipeline.retriever.retrieve(
                query=body.question,
                k=body.k * 2,
                filter_metadata=filter_metadata,
                use_hybrid=body.use_hybrid,
            )

            if not retrieved:
                yield sse_event("token", {
                    "content": "No relevant information found in the uploaded documents."
                })
                yield sse_event("sources", {"sources": []})
                yield sse_event("done", {})
                return

            # Rerank
            reranked = pipeline.reranker.rerank(body.question, retrieved, top_k=body.k)
            yield sse_event("status", {
                "message": f"Found {len(reranked)} relevant sections. Generating answer..."
            })

            # Try streaming from Groq
            import os
            groq_key = os.getenv("GROQ_API_KEY", "")
            streamed = False

            if groq_key and pipeline.llm_router.active_provider == "groq":
                try:
                    from groq import Groq
                    client = Groq(api_key=groq_key)

                    # Build context
                    context = pipeline.generator._build_context(reranked)
                    prompt = f"""{pipeline.llm_router.FINANCIAL_SYSTEM_PROMPT}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{body.question}

ANSWER (with citations):"""

                    conv_context = pipeline._get_conversation_context(body.session_id)
                    if conv_context:
                        prompt = f"Previous context: {conv_context}\n\n{prompt}"

                    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                    stream = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048,
                        temperature=0.3,
                        stream=True,
                    )

                    full_answer = ""
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            full_answer += token
                            yield sse_event("token", {"content": token})

                    streamed = True

                except Exception as e:
                    logger.warning(f"Streaming failed, falling back: {e}")

            # Fallback: non-streaming generation
            if not streamed:
                conv_context = pipeline._get_conversation_context(body.session_id)
                result = pipeline.generator.generate(
                    query=body.question,
                    retrieved_chunks=reranked,
                    conversation_context=conv_context,
                )
                full_answer = result["answer"]
                yield sse_event("token", {"content": full_answer})

            # Extract sources
            sources = pipeline.generator._extract_sources(reranked)
            yield sse_event("sources", {"sources": sources})

            # Save conversation and cache
            pipeline._update_conversation(
                body.session_id, body.question, full_answer,
                sources, {"provider": "groq" if streamed else "fallback"},
            )

            try:
                response_dict = {"answer": full_answer, "sources": sources, "metadata": {}}
                pipeline.semantic_cache.set(body.question, query_embedding, response_dict)
            except Exception:
                pass

            yield sse_event("done", {})

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}")
async def get_history(session_id: str, request: Request):
    """Get conversation history for a session."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if not pipeline:
        return {"history": []}

    history = pipeline.get_conversation_history(session_id)
    return {"session_id": session_id, "history": history}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str, request: Request):
    """Clear conversation history for a session."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if pipeline:
        pipeline.clear_conversation(session_id)

    return {"success": True, "message": f"Session '{session_id}' cleared"}


@router.get("/evaluate")
async def evaluate_pipeline(request: Request, suite: str = "financial_qa"):
    """Run evaluation suite against the RAG pipeline."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        from src.evaluation.evaluator import RAGEvaluator
        from src.evaluation.benchmarks import BenchmarkSuite

        evaluator = RAGEvaluator(rag_pipeline=pipeline)
        
        # Load benchmark suite
        suites = BenchmarkSuite.get_all_suites()
        test_cases = suites.get(suite, [])
        
        if not test_cases:
            return {"success": False, "error": f"Benchmark suite '{suite}' not found"}

        # Run evaluation
        results = evaluator.evaluate(test_cases, verbose=False)
        
        return {
            "success": True, 
            "suite": suite,
            "metrics": results.get("aggregate_metrics", {}),
            "summary": {
                "total": results.get("total_test_cases", 0),
                "avg_latency": results.get("avg_latency_seconds", 0),
            },
            "detailed_results": results.get("per_query_results", [])
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"success": False, "error": str(e)}
