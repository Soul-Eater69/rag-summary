"""
Canonical data contract for summary documents.

All summary dicts flowing through the pipeline — whether generated from
historical tickets or from new idea cards — must conform to this schema.

Usage
-----
From other modules::

    from summary_rag.ingestion.schema import SummaryDoc

Type hints::

    def build_retrieval_text(summary: SummaryDoc) -> str: ...
"""

from __future__ import annotations

from typing import List
from typing_extensions import NotRequired, TypedDict


class SummaryDoc(TypedDict):
    """One summary document, stored in the FAISS index and returned by retrieval."""

    # Stable identifiers
    doc_id: str          # e.g. "summary_IDMT-19761" or "new_idea_card_summary"
    ticket_id: str       # source ticket ID; empty string for new-card summaries
    title: str           # human-readable title; empty string if unknown

    # Core semantic fields (LLM-generated or deterministic fallback)
    short_summary: str
    business_goal: str
    actors: List[str]
    direct_functions: List[str]
    implied_functions: List[str]
    change_types: List[str]
    domain_tags: List[str]
    evidence_sentences: List[str]

    # Ground-truth labels (historical tickets only; empty list for new cards)
    value_stream_labels: List[str]

    # Packed retrieval text — persisted alongside the doc so callers never
    # have to rebuild it from scratch.  Populated by build_retrieval_text().
    retrieval_text: NotRequired[str]
