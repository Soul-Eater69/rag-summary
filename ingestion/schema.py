"""
Canonical data contract for summary documents (V5 architecture).

All summary dicts flowing through the pipeline -- whether generated from
historical tickets or from new idea cards -- must conform to this schema.

V5 changes:
- Separate raw vs canonical for direct/implied functions
- Added capability_tags, operational_footprint
- Added stream_support_type, supporting_evidence
- Added co_occurrence_bundle for pattern inference

Usage
-----
From other modules::

    from summary_rag.ingestion.schema import SummaryDoc

Type hints::

    def build_retrieval_text(summary: SummaryDoc) -> str: ...
"""

from __future__ import annotations

from typing import Dict, List
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
    change_types: List[str]
    domain_tags: List[str]
    evidence_sentences: List[str]

    # V5: raw LLM-extracted functions (free-form)
    direct_functions_raw: List[str]
    implied_functions_raw: List[str]

    # V5: normalized to canonical vocabulary
    direct_functions_canonical: List[str]
    implied_functions_canonical: List[str]

    # Legacy compat aliases (populated from canonical)
    direct_functions: List[str]
    implied_functions: List[str]

    # V5: capability and operational metadata
    capability_tags: NotRequired[List[str]]
    operational_footprint: NotRequired[List[str]]

    # Ground-truth labels (historical tickets only; empty list for new cards)
    value_stream_labels: List[str]
    value_stream_ids: NotRequired[List[str]]

    # V5: per-stream support classification
    stream_support_type: NotRequired[Dict[str, str]]

    # V5: short evidence snippets justifying major streams
    supporting_evidence: NotRequired[List[str]]

    # V5: compact co-occurrence bundle for pattern inference
    co_occurrence_bundle: NotRequired[List[str]]

    # Packed retrieval text -- persisted alongside the doc so callers never
    # have to rebuild it from scratch.  Populated by build_retrieval_text().
    retrieval_text: NotRequired[str]
