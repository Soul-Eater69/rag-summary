"""
LangGraph PredictionState (V5 architecture).

TypedDict used as the LangGraph state object, passing data between graph nodes.
Using TypedDict (not Pydantic BaseModel) for LangGraph compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class PredictionState(TypedDict, total=False):
    """
    Shared mutable state flowing through the LangGraph prediction pipeline.

    All fields are optional (total=False) so nodes can be added incrementally.
    """

    # --- Inputs ---
    raw_text: str                           # Raw idea card text
    cleaned_text: str                       # Cleaned card text
    allowed_value_stream_names: List[str]   # Allow-list of VS names
    top_k_analogs: int                      # How many analogs to retrieve

    # --- Step 1: Card summarization ---
    new_card_summary: Dict[str, Any]        # SummaryDoc-shaped dict from LLM

    # --- Step 2: FAISS analog retrieval ---
    analog_tickets: List[Dict[str, Any]]    # Top-K analog SummaryDocs from FAISS

    # --- Step 3: VS evidence from analogs ---
    historical_value_stream_support: List[Dict[str, Any]]

    # --- Step 4: KG candidates ---
    kg_candidates: List[Dict[str, Any]]

    # --- Step 5: Capability mapping output ---
    capability_mapping: Dict[str, Any]
    enriched_candidates: List[Dict[str, Any]]

    # --- Step 4b: Theme candidates (live if ThemeRetrievalService is wired) ---
    theme_candidates: List[Dict[str, Any]]

    # --- Step 5b: Card-level candidates ---
    summary_candidates: List[Dict[str, Any]]
    chunk_candidates: List[Dict[str, Any]]
    card_attachment_candidates: List[Dict[str, Any]]  # card-native attachment signals
    historical_footprint_candidates: List[Dict[str, Any]]

    # --- Step 6: Raw evidence (for attachment proxy from analogs) ---
    raw_evidence: List[Dict[str, Any]]
    attachment_candidates: List[Dict[str, Any]]

    # --- Step 7: CandidateEvidence (pre-fusion) ---
    candidate_evidence: List[Dict[str, Any]]

    # --- Step 8: Fused scores ---
    fused_candidates: List[Dict[str, Any]]

    # --- Step 9: LLM verification / selection ---
    verify_judgments: List[Dict[str, Any]]  # List[CandidateJudgment] dicts from Pass 1
    selection_result: Dict[str, Any]        # SelectionResult-shaped dict from Pass 2

    # --- Final output ---
    directly_supported: List[Dict[str, Any]]
    pattern_inferred: List[Dict[str, Any]]
    no_evidence: List[Dict[str, Any]]
    selected_value_streams: List[Dict[str, Any]]  # compat union
    rejected_candidates: List[Dict[str, Any]]

    # --- Diagnostics ---
    errors: List[str]
    warnings: List[str]
    timing: Dict[str, float]
