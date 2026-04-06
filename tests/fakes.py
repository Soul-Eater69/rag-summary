"""
Fake service implementations for testing (V6).

Drop-in replacements for all external service dependencies.
Each fake is configured with fixtures at construction time, so tests
can control exactly what the pipeline sees without touching real infra.

Usage:
    from tests.fakes import (
        FakeLLM, FakeEmbedding, FakeKG, FakeTheme,
        FakeSummaryIndex, FakeRawEvidence,
    )
    from rag_summary.graph.service_container import ServiceContainer

    container = ServiceContainer(
        llm=FakeLLM(response='{"short_summary": "stub summary"}'),
        embedding=FakeEmbedding(dim=4),
        kg=FakeKG(candidates=[{"entity_name": "VS-A", "score": 0.8}]),
        theme=FakeTheme(candidates=[{"entity_name": "VS-B", "score": 0.6, "source": "theme"}]),
        summary_index=FakeSummaryIndex(hits=[{"ticket_id": "T-1", "score": 0.9, "value_stream_labels": ["VS-A"]}]),
        raw_evidence=FakeRawEvidence(chunks={"T-1": [{"text": "some chunk text", "score": 0.7}]}),
    )
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class FakeLLM:
    """
    Fake LLMService and StructuredLLMService.

    Returns a fixed response string. If pydantic_response is provided,
    generate_structured() returns it directly (bypassing JSON parsing).
    """

    def __init__(
        self,
        response: str = "{}",
        pydantic_response: Any = None,
    ) -> None:
        self._response = response
        self._pydantic_response = pydantic_response

    def generate(
        self,
        query: str,
        *,
        context: str = "",
        system_prompt: str = "",
    ) -> SimpleNamespace:
        return SimpleNamespace(content=self._response)

    def generate_structured(
        self,
        query: str,
        output_schema: type,
        *,
        context: str = "",
        system_prompt: str = "",
    ) -> Any:
        if self._pydantic_response is not None:
            return self._pydantic_response
        # Fall back: parse self._response as the schema
        import json
        try:
            data = json.loads(self._response)
            return output_schema.model_validate(data)
        except Exception:
            return output_schema()


class FakeEmbedding:
    """
    Fake EmbeddingService.

    Returns deterministic unit vectors so FAISS distances are predictable.
    embed_query returns the same fixed vector for every query.
    embed_documents returns one vector per document.
    """

    def __init__(self, dim: int = 8, value: float = 0.5) -> None:
        self._dim = dim
        self._value = value

    def _unit_vec(self) -> List[float]:
        import math
        v = [self._value] * self._dim
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_query(self, text: str) -> List[float]:
        return self._unit_vec()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._unit_vec() for _ in texts]


class FakeKG:
    """
    Fake KGRetrievalService.

    Returns a fixed list of candidate dicts, optionally filtered by allowed_names.
    """

    def __init__(self, candidates: Optional[List[Dict[str, Any]]] = None) -> None:
        self._candidates = candidates or []

    def retrieve_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 20,
        allowed_names: Optional[List[str]] = None,
    ) -> List[dict]:
        results = self._candidates[:top_k]
        if allowed_names is not None:
            allowed = set(allowed_names)
            results = [c for c in results if c.get("entity_name", "") in allowed]
        return results


class FakeTheme:
    """
    Fake ThemeRetrievalService.

    Returns a fixed list of theme candidate dicts.
    """

    def __init__(self, candidates: Optional[List[Dict[str, Any]]] = None) -> None:
        self._candidates = candidates or []

    def retrieve_theme_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        allowed_names: Optional[List[str]] = None,
        cutoff_date: Optional[str] = None,
    ) -> List[dict]:
        results = self._candidates[:top_k]
        if allowed_names is not None:
            allowed = set(allowed_names)
            results = [c for c in results if c.get("entity_name", "") in allowed]
        return results


class FakeSummaryIndex:
    """
    Fake SummaryIndexService.

    Returns a fixed list of analog ticket hit dicts.
    """

    def __init__(self, hits: Optional[List[Dict[str, Any]]] = None) -> None:
        self._hits = hits or []

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        allowed_vs_names: Optional[List[str]] = None,
    ) -> List[dict]:
        return self._hits[:top_k]


class FakeRawEvidence:
    """
    Fake RawEvidenceService.

    Returns pre-configured chunks keyed by ticket_id.
    Falls back to empty list for unknown ticket_ids.
    """

    def __init__(self, chunks: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
        self._chunks = chunks or {}

    def get_chunks_for_ticket(
        self,
        ticket_id: str,
        *,
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> List[dict]:
        return self._chunks.get(ticket_id, [])[:top_k]
