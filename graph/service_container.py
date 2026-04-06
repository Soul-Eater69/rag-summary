"""
ServiceContainer — dependency injection container for the prediction pipeline (V6).

All external service dependencies are gathered into a single dataclass so nodes
can accept a container rather than individual keyword arguments, making tests
easy to write by swapping in fake implementations.

Usage:
    from rag_summary.graph.service_container import ServiceContainer, build_default_container

    # Production
    container = build_default_container(
        index_dir="path/to/faiss_index",
        ticket_chunks_dir="ticket_chunks",
        theme_index_dir="config/theme_index",
    )

    # Test
    from tests.fakes import FakeSummaryIndex, FakeKG
    container = ServiceContainer(
        summary_index=FakeSummaryIndex(hits=[...]),
        kg=FakeKG(candidates=[...]),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rag_summary.ingestion.adapters import (
    EmbeddingService,
    KGRetrievalService,
    LLMService,
    RawEvidenceService,
    SummaryIndexService,
    ThemeRetrievalService,
    get_default_embedding,
    get_default_kg,
    get_default_llm,
    get_default_theme,
)


@dataclass
class ServiceContainer:
    """
    Holds all external service dependencies for the prediction pipeline.

    Fields left as None fall back to the default factory (lazy-initialized
    from src.*) when accessed via build_default_container(). Tests can
    inject fakes for any subset of services.
    """

    llm: Optional[LLMService] = None
    embedding: Optional[EmbeddingService] = None
    kg: Optional[KGRetrievalService] = None
    theme: Optional[ThemeRetrievalService] = None
    summary_index: Optional[SummaryIndexService] = None
    raw_evidence: Optional[RawEvidenceService] = None

    # Carry-through config for node resolution (mirrors _private state keys)
    index_dir: str = "faiss_index"
    ticket_chunks_dir: str = "ticket_chunks"
    intake_date: Optional[str] = None


def build_default_container(
    *,
    index_dir: str = "faiss_index",
    ticket_chunks_dir: str = "ticket_chunks",
    theme_index_dir: Optional[str] = None,
    intake_date: Optional[str] = None,
    llm: Optional[LLMService] = None,
    embedding: Optional[EmbeddingService] = None,
    kg: Optional[KGRetrievalService] = None,
) -> ServiceContainer:
    """
    Build a ServiceContainer with default production implementations.

    Services that require optional dependencies (faiss, theme index files)
    degrade gracefully if those deps are absent.

    Args:
        index_dir:        Path to FAISS summary index directory.
        ticket_chunks_dir: Path to ticket_chunks/ directory.
        theme_index_dir:  Path to theme FAISS index directory.  If None,
                          the noop theme service is used.
        intake_date:      ISO-8601 date string for temporal leakage prevention.
        llm:              Override LLM service (default: get_default_llm()).
        embedding:        Override embedding service (default: get_default_embedding()).
        kg:               Override KG service (default: get_default_kg()).
    """
    resolved_llm = llm or get_default_llm()
    resolved_embedding = embedding or get_default_embedding()
    resolved_kg = kg or get_default_kg()

    # Theme service: use real FAISS impl if theme_index_dir provided
    if theme_index_dir:
        from rag_summary.ingestion.theme_retrieval_service import FaissThemeRetrievalService
        resolved_theme: ThemeRetrievalService = FaissThemeRetrievalService(
            theme_index_dir=theme_index_dir,
            embedding_svc=resolved_embedding,
        )
    else:
        resolved_theme = get_default_theme()

    # Summary index: wrap FaissIndexer in adapter
    from rag_summary.ingestion.adapters_impl import FaissIndexAdapter
    resolved_summary_index: SummaryIndexService = FaissIndexAdapter(
        index_dir=index_dir,
        embedding_svc=resolved_embedding,
    )

    # Raw evidence: wrap filesystem ticket_chunks/ reader
    from rag_summary.ingestion.adapters_impl import TicketChunksAdapter
    resolved_raw_evidence: RawEvidenceService = TicketChunksAdapter(
        ticket_chunks_dir=ticket_chunks_dir,
    )

    return ServiceContainer(
        llm=resolved_llm,
        embedding=resolved_embedding,
        kg=resolved_kg,
        theme=resolved_theme,
        summary_index=resolved_summary_index,
        raw_evidence=resolved_raw_evidence,
        index_dir=index_dir,
        ticket_chunks_dir=ticket_chunks_dir,
        intake_date=intake_date,
    )
