"""
Adapter layer (V5 architecture).

Thin Protocol interfaces for the three external service dependencies:
  - LLMService: text generation (wraps src.services.generation_service)
  - EmbeddingService: vector embeddings (wraps src.clients.embedding)
  - KGRetrievalService: knowledge-graph candidate retrieval (wraps
    src.pipelines.value_stream.retrieval_pipeline)

Usage
-----
Modules should accept an optional adapter parameter::

    def generate_new_card_summary(
        ppt_text: str,
        *,
        llm: LLMService | None = None,
    ) -> SummaryDoc:
        svc = llm or get_default_llm()
        ...

Tests can inject stub implementations without touching src.*::

    class StubLLM:
        def generate(self, query, *, context="", system_prompt=""):
            return SimpleNamespace(content='{"short_summary": "stub"}')

    summary = generate_new_card_summary(text, llm=StubLLM())

The factory functions (get_default_*) lazy-import from src.* so the
adapters module itself is importable even when src.* is absent.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocols (structural interfaces)
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMResponse(Protocol):
    """Minimal contract for an LLM response object."""
    content: str


@runtime_checkable
class LLMService(Protocol):
    """Text generation service interface."""

    def generate(
        self,
        query: str,
        *,
        context: str = "",
        system_prompt: str = "",
    ) -> LLMResponse:
        ...


@runtime_checkable
class EmbeddingService(Protocol):
    """Vector embedding service interface."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


@runtime_checkable
class KGRetrievalService(Protocol):
    """Knowledge-graph candidate retrieval interface."""

    def retrieve_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 20,
        allowed_names: Optional[List[str]] = None,
    ) -> List[dict]:
        ...


# ---------------------------------------------------------------------------
# Default factory functions (lazy import from src.*)
# ---------------------------------------------------------------------------

def get_default_llm() -> LLMService:
    """Return the default LLM service from the internal codebase."""
    from src.services.generation_service import GenerationService  # type: ignore[import]
    return GenerationService()


def get_default_embedding() -> EmbeddingService:
    """Return the default embedding client from the internal codebase."""
    from src.clients.embedding import EmbeddingClient  # type: ignore[import]
    return EmbeddingClient()


class _KGRetrievalAdapter:
    """Thin adapter wrapping the internal KG retrieval pipeline."""

    def retrieve_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 20,
        allowed_names: Optional[List[str]] = None,
    ) -> List[dict]:
        from src.pipelines.value_stream.retrieval_pipeline import (  # type: ignore[import]
            retrieve_kg_candidates as _retrieve,
        )
        return _retrieve(query_text, top_k=top_k, allowed_names=allowed_names)


def get_default_kg() -> KGRetrievalService:
    """Return the default KG retrieval adapter."""
    return _KGRetrievalAdapter()


# ---------------------------------------------------------------------------
# Utility: safe JSON extraction (wraps core.prompts)
# ---------------------------------------------------------------------------

def safe_json_extract(text: str) -> Any:
    """
    Parse JSON from an LLM response string.

    Tries core.prompts.safe_json_extract first; falls back to a simple
    inline implementation so the adapter module is usable without core.*.
    """
    try:
        from core.prompts import safe_json_extract as _core_extract  # type: ignore[import]
        return _core_extract(text)
    except Exception:
        pass
    # Fallback: find the first {...} or [...] block and parse it
    import json
    import re
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def normalize_text(text: str) -> str:
    """
    Lowercase, strip, and normalize text for search/comparison.

    Tries core.text.normalize_for_search first; falls back to simple
    lower+strip so the module is usable without core.*.
    """
    try:
        from core.text import normalize_for_search  # type: ignore[import]
        return normalize_for_search(text or "")
    except Exception:
        return (text or "").lower().strip()


def clean_card_text(text: str) -> str:
    """
    Clean/normalize raw PPT / idea card text.

    Tries core.text.clean_ppt_text first; falls back to basic whitespace
    normalization.
    """
    try:
        from core.text import clean_ppt_text  # type: ignore[import]
        return clean_ppt_text(text or "")
    except Exception:
        import re
        return re.sub(r"\s+", " ", (text or "")).strip()
