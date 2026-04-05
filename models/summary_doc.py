"""
Pydantic SummaryDoc model (V5 architecture).

Replaces the TypedDict in ingestion/schema.py as the canonical data contract.
All summary dicts flowing through the pipeline conform to this schema.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class SummaryDoc(BaseModel):
    """One summary document, stored in the FAISS index and returned by retrieval."""

    # Stable identifiers
    doc_id: str = ""
    ticket_id: str = ""
    title: str = ""

    # Core semantic fields
    short_summary: str = ""
    business_goal: str = ""
    actors: List[str] = Field(default_factory=list)
    change_types: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)
    evidence_sentences: List[str] = Field(default_factory=list)

    # V5: raw LLM-extracted functions (free-form)
    direct_functions_raw: List[str] = Field(default_factory=list)
    implied_functions_raw: List[str] = Field(default_factory=list)

    # V5: normalized to canonical vocabulary
    direct_functions_canonical: List[str] = Field(default_factory=list)
    implied_functions_canonical: List[str] = Field(default_factory=list)

    # Legacy compat aliases (populated from canonical)
    direct_functions: List[str] = Field(default_factory=list)
    implied_functions: List[str] = Field(default_factory=list)

    # V5: capability and operational metadata
    capability_tags: List[str] = Field(default_factory=list)
    operational_footprint: List[str] = Field(default_factory=list)

    # Ground-truth labels (historical tickets only; empty for new cards)
    value_stream_labels: List[str] = Field(default_factory=list)
    value_stream_ids: List[str] = Field(default_factory=list)

    # V5: per-stream support classification
    stream_support_type: Dict[str, str] = Field(default_factory=dict)

    # V5: short evidence snippets justifying major streams
    supporting_evidence: List[str] = Field(default_factory=list)

    # V5: compact co-occurrence bundle for pattern inference
    co_occurrence_bundle: List[str] = Field(default_factory=list)

    # Packed retrieval text
    retrieval_text: str = ""

    class Config:
        extra = "allow"
