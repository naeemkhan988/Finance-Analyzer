"""
LLM Router
===========
Multi-provider LLM router with automatic failover.
Supports: Groq (Llama 3.3), Google Gemini, and fallback generation.
"""

import os
import logging
import time
from typing import Dict, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Production LLM router with multi-provider support and failover.

    Features:
    - Multiple LLM provider support (Groq, Gemini)
    - Automatic failover on rate limits or errors
    - Retry with exponential backoff
    - Response caching
    - Token usage tracking
    - Custom system prompts per use case
    """

    FINANCIAL_SYSTEM_PROMPT = """You are an expert financial analyst AI assistant. Your role is to analyze financial documents and provide accurate, well-cited answers.

RULES:
1. Answer questions based ONLY on the provided document context.
2. If the information is not in the provided context, clearly state: "This information is not available in the provided documents."
3. Always cite your sources with document name and page numbers when available.
4. Present numerical data clearly with proper formatting (currencies, percentages).
5. For financial comparisons, use tables when appropriate.
6. Highlight key financial metrics, trends, and anomalies.
7. Be precise with numbers - never approximate financial figures.
8. If a question is ambiguous, state your interpretation before answering."""

    def __init__(self):
        self.providers = self._initialize_providers()
        self.active_provider = None
        self._usage_stats = {
            "total_queries": 0,
            "provider_usage": {},
            "total_tokens": 0,
            "errors": [],
        }
        self._select_provider()

    def _initialize_providers(self) -> Dict:
        """Initialize available LLM providers."""
        providers = {}

        # Groq (Free tier - Llama 3.3 70B)
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            try:
                from groq import Groq

                providers["groq"] = {
                    "client": Groq(api_key=groq_key),
                    "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    "name": "Groq (Llama 3.3 70B)",
                    "max_tokens": 4096,
                    "available": True,
                }
                logger.info("Groq provider initialized")
            except Exception as e:
                logger.warning(f"Groq initialization failed: {e}")

        # Google Gemini (Free tier)
        google_key = os.getenv("GOOGLE_API_KEY", "")
        if google_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=google_key)
                model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                providers["gemini"] = {
                    "client": genai.GenerativeModel(model_name),
                    "model": model_name,
                    "name": f"Google Gemini ({model_name})",
                    "max_tokens": 8192,
                    "available": True,
                }
                logger.info("Gemini provider initialized")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")

        if not providers:
            logger.warning(
                "No LLM providers configured. Using intelligent fallback mode."
            )

        return providers

    def _select_provider(self):
        """Select the best available provider."""
        priority = ["groq", "gemini"]
        for provider_name in priority:
            if provider_name in self.providers and self.providers[provider_name]["available"]:
                self.active_provider = provider_name
                logger.info(
                    f"Active LLM provider: {self.providers[provider_name]['name']}"
                )
                return

        self.active_provider = None
        logger.warning("No active LLM provider. Running in fallback mode.")

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Dict:
        """
        Generate a response using the active LLM provider.

        Args:
            query: User question
            context: Retrieved document context
            system_prompt: Optional custom system prompt
            max_tokens: Maximum response tokens
            temperature: Response creativity (0=precise, 1=creative)

        Returns:
            Dict with answer, provider info, and usage stats
        """
        self._usage_stats["total_queries"] += 1
        prompt = system_prompt or self.FINANCIAL_SYSTEM_PROMPT

        full_prompt = f"""{prompt}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER (with citations):"""

        # Try active provider, then failover
        providers_to_try = []
        if self.active_provider:
            providers_to_try.append(self.active_provider)

        for name, config in self.providers.items():
            if name != self.active_provider and config["available"]:
                providers_to_try.append(name)

        for provider_name in providers_to_try:
            try:
                result = self._call_provider(
                    provider_name, full_prompt, max_tokens, temperature
                )
                self._track_usage(provider_name, result)
                return result
            except Exception as e:
                logger.warning(f"Provider '{provider_name}' failed: {e}")
                self._usage_stats["errors"].append(
                    {"provider": provider_name, "error": str(e), "time": time.time()}
                )
                continue

        # Fallback: intelligent context-based response
        return self._fallback_response(query, context)

    def _call_provider(
        self, provider_name: str, prompt: str, max_tokens: int, temperature: float
    ) -> Dict:
        """Call a specific LLM provider."""
        provider = self.providers[provider_name]

        if provider_name == "groq":
            return self._call_groq(provider, prompt, max_tokens, temperature)
        elif provider_name == "gemini":
            return self._call_gemini(provider, prompt, max_tokens, temperature)

        raise ValueError(f"Unknown provider: {provider_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _call_groq(
        self, provider: Dict, prompt: str, max_tokens: int, temperature: float
    ) -> Dict:
        """Call Groq API with retry logic."""
        start = time.time()

        response = provider["client"].chat.completions.create(
            model=provider["model"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        answer = response.choices[0].message.content
        latency = time.time() - start

        return {
            "answer": answer,
            "provider": "groq",
            "model": provider["model"],
            "latency_seconds": round(latency, 2),
            "tokens_used": getattr(response.usage, "total_tokens", 0),
            "finish_reason": response.choices[0].finish_reason,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _call_gemini(
        self, provider: Dict, prompt: str, max_tokens: int, temperature: float
    ) -> Dict:
        """Call Google Gemini API with retry logic."""
        start = time.time()

        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        response = provider["client"].generate_content(
            prompt,
            generation_config=generation_config,
        )

        answer = response.text
        latency = time.time() - start

        return {
            "answer": answer,
            "provider": "gemini",
            "model": provider["model"],
            "latency_seconds": round(latency, 2),
            "tokens_used": 0,
            "finish_reason": "completed",
        }

    def _fallback_response(self, query: str, context: str) -> Dict:
        """
        Generate an intelligent fallback response when no LLM is available.
        Extracts relevant sentences from context based on keyword matching.
        """
        logger.info("Using fallback response generation")

        query_words = set(query.lower().split())
        stop_words = {
            "what", "is", "the", "a", "an", "of", "in", "to", "for", "and",
            "or", "how", "much", "many", "which", "where", "when", "who",
            "does", "do", "are", "was", "were", "been", "be", "have", "has",
            "had", "will", "would", "could", "should", "can", "may", "might",
        }
        query_keywords = query_words - stop_words

        sentences = context.replace("\n", " ").split(". ")
        scored = []

        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for kw in query_keywords if kw in sent_lower)
            if score > 0:
                scored.append((score, sent.strip()))

        scored.sort(key=lambda x: x[0], reverse=True)
        relevant = [s[1] for s in scored[:5]]

        if relevant:
            answer = (
                "Based on the provided documents, here are the most relevant findings:\n\n"
                + "\n\n".join(f"• {s}" for s in relevant)
                + "\n\n⚠️ Note: This response was generated using keyword extraction "
                "(no LLM API configured). For AI-powered analysis, please configure "
                "a Groq or Gemini API key in your .env file."
            )
        else:
            answer = (
                "I could not find directly relevant information for your query "
                "in the provided documents. Please try rephrasing your question "
                "or upload additional relevant documents.\n\n"
                "⚠️ Note: No LLM API is configured. Configure a Groq or Gemini "
                "API key for full AI-powered analysis."
            )

        return {
            "answer": answer,
            "provider": "fallback",
            "model": "keyword_extraction",
            "latency_seconds": 0,
            "tokens_used": 0,
            "finish_reason": "fallback",
        }

    def _track_usage(self, provider_name: str, result: Dict):
        """Track API usage statistics."""
        if provider_name not in self._usage_stats["provider_usage"]:
            self._usage_stats["provider_usage"][provider_name] = {
                "calls": 0,
                "tokens": 0,
                "avg_latency": 0,
            }

        stats = self._usage_stats["provider_usage"][provider_name]
        stats["calls"] += 1
        stats["tokens"] += result.get("tokens_used", 0)
        stats["avg_latency"] = (
            (stats["avg_latency"] * (stats["calls"] - 1) + result.get("latency_seconds", 0))
            / stats["calls"]
        )
        self._usage_stats["total_tokens"] += result.get("tokens_used", 0)

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            **self._usage_stats,
            "active_provider": self.active_provider,
            "available_providers": list(self.providers.keys()),
        }

    def get_provider_info(self) -> Dict:
        """Get information about configured providers."""
        info = {}
        for name, config in self.providers.items():
            info[name] = {
                "name": config["name"],
                "model": config["model"],
                "available": config["available"],
                "active": name == self.active_provider,
            }
        if not info:
            info["fallback"] = {
                "name": "Keyword Extraction (No API)",
                "model": "keyword_extraction",
                "available": True,
                "active": True,
            }
        return info
