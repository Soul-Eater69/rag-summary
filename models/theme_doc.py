"""
Theme document models (V6 architecture).

ThemeDoc represents a cluster of historically co-occurring value streams.
Built offline by tools/build_theme_index.py and searched at runtime by
ingestion/theme_retrieval_service.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ThemeDoc(BaseModel):
    """
    One theme cluster, built from historical tickets sharing similar VS patterns.

    A theme captures the idea that certain VS combinations appear together
    repeatedly across historical initiatives. When a new card matches a theme,
    all VS members get a calibrated boost proportional to vs_support_fraction.
    """
    theme_id: str                            # stable UUID assigned at cluster creation
    theme_label: str                         # human-readable label e.g. "care-management-workflow"
    theme_description: str = ""             # 2-3 sentence summary of the pattern
    member_ticket_ids: List[str] = Field(default_factory=list)
    member_count: int = 0

    # VS associations: which streams this theme predicts and how strongly
    value_stream_names: List[str] = Field(default_factory=list)
    vs_support_counts: Dict[str, int] = Field(default_factory=dict)
    vs_support_fractions: Dict[str, float] = Field(default_factory=dict)

    # Semantic content for retrieval and debugging
    canonical_functions: List[str] = Field(default_factory=list)
    capability_tags: List[str] = Field(default_factory=list)
    cue_phrases: List[str] = Field(default_factory=list)
    retrieval_text: str = ""                # packed text used for embedding

    # Temporal safety: themes built only from pre-intake historical tickets
    last_ticket_ingested_at: str = ""       # ISO-8601; latest ticket timestamp in cluster
    created_at: str = ""
    updated_at: str = ""

    # Quality
    cohesion_score: float = 0.0             # mean cosine similarity to centroid; 0–1
    min_vs_support_fraction: float = 0.30   # minimum fraction to include a VS in candidates

    class Config:
        extra = "allow"


class ThemeIndexManifest(BaseModel):
    """Manifest written alongside theme_index.faiss and theme_docs.json."""
    built_at: str
    source_faiss_dir: str
    source_ticket_count: int
    theme_count: int
    discarded_clusters: int = 0             # clusters below cohesion threshold
    vs_coverage: Dict[str, int] = Field(default_factory=dict)   # VS name → theme count
    uncovered_vs_names: List[str] = Field(default_factory=list)
    build_params: Dict[str, Any] = Field(default_factory=dict)
