"""
Pydantic CandidateEvidence model (V5 architecture).

CandidateEvidence is the core runtime artifact: one per candidate value stream,
with multi-source provenance, scores, snippets, and support type classification.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


# Source keys
SOURCE_CHUNK = "chunk"
SOURCE_SUMMARY = "summary"
SOURCE_ATTACHMENT = "attachment"
SOURCE_THEME = "theme"
SOURCE_KG = "kg"
SOURCE_HISTORICAL = "historical"
SOURCE_CAPABILITY = "capability"

ALL_SOURCES = [
    SOURCE_CHUNK,
    SOURCE_SUMMARY,
    SOURCE_ATTACHMENT,
    SOURCE_THEME,
    SOURCE_KG,
    SOURCE_HISTORICAL,
    SOURCE_CAPABILITY,
]


def _default_source_scores() -> Dict[str, float]:
    return {s: 0.0 for s in ALL_SOURCES}


class EvidenceSnippet(BaseModel):
    source: str
    snippet: str


class CandidateEvidence(BaseModel):
    candidate_id: str = ""
    candidate_name: str
    description: str = ""
    source_scores: Dict[str, float] = Field(default_factory=_default_source_scores)
    evidence_sources: List[str] = Field(default_factory=list)
    evidence_snippets: List[EvidenceSnippet] = Field(default_factory=list)
    fused_score: float = 0.0
    support_confidence: float = 0.0
    source_diversity_count: int = 0
    support_type: str = "none"
    contradictions: List[str] = Field(default_factory=list)

    class Config:
        extra = "allow"
