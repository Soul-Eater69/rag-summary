"""
LangGraph PredictionState (V6 architecture).

TypedDict used as the LangGraph state object, passing data between graph nodes.
Using TypedDict (not Pydantic BaseModel) for LangGraph compatibility.

14-node graph:
  clean_and_summarize -> retrieve_analogs -> collect_vs_evidence
  -> retrieve_kg -> retrieve_themes -> map_capabilities
  -> extract_card_candidates -> collect_raw_evidence
  -> parse_attachments -> build_evidence -> fuse_scores
  -> verify_candidates -> finalize_selection -> finalize_output

7 evidence sources:
  chunk | summary | attachment | theme | kg | historical | capability
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

class PredictionState(TypedDict, total=False):
    """
    Shared mutable state flowing through the LangGraph prediction pipeline.

    All fields are optional (total=False) so nodes can be added incrementally.
    Private config keys (_*) are injected by run_prediction_graph() at startup
    and use type: ignore at injection sites since TypedDict disallows unknown keys.
    """

    # ---------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------
    raw_text: str                            # Raw idea card text
    cleaned_text: str                        # Cleaned, normalized card text
    allowed_value_stream_names: list[str]    # Allow-list of VS names (empty = all)
    top_k_analogs: int                       # How many FAISS analogs to retrieve

    # ---------------------------------------------------------
    # Step 1 - clean_and_summarize
    # ---------------------------------------------------------
    new_card_summary: Dict[str, Any]         # SummaryDoc-shaped dict from LLM
    summary_prompt: Dict[str, Any]           # Prompt payload for summary generation
    summary_debug: Dict[str, Any]            # Raw response / parse diagnostics for summary generation

    # ---------------------------------------------------------
    # Step 2 - retrieve_analogs
    # ---------------------------------------------------------
    analog_tickets: List[Dict[str, Any]]     # Top-K analog SummaryDocs from FAISS

    # ---------------------------------------------------------
    # Step 3 - collect_vs_evidence
    # ---------------------------------------------------------
    historical_value_stream_support: List[Dict[str, Any]] # VS support from analogs
    bundle_patterns: List[Dict[str, Any]]                 # VS pairs co-occurring >=60% across analogs
    downstream_chains: List[Dict[str, Any]]               # VS downstream-activation chains

    # ---------------------------------------------------------
    # Step 4 - retrieve_kg
    # ---------------------------------------------------------
    kg_candidates: List[Dict[str, Any]]

    # ---------------------------------------------------------
    # Step 4b - retrieve_themes
    # ---------------------------------------------------------
    theme_candidates: List[Dict[str, Any]]
    theme_source_status: Dict[str, Any]      # backend, active, candidate_count, max_score
    theme_debug: Dict[str, Any]              # per-candidate debug detail

    # ---------------------------------------------------------
    # Step 5 - map_capabilities
    # ---------------------------------------------------------
    capability_mapping: Dict[str, Any]
    enriched_candidates: List[Dict[str, Any]]

    # ---------------------------------------------------------
    # Step 5b - extract_card_candidates
    # ---------------------------------------------------------
    summary_candidates: List[Dict[str, Any]]
    chunk_candidates: List[Dict[str, Any]]
    card_attachment_candidates: List[Dict[str, Any]]  # sub_source="attachment_heuristic"
    historical_footprint_candidates: List[Dict[str, Any]]

    # ---------------------------------------------------------
    # Step 6 - collect_raw_evidence
    # ---------------------------------------------------------
    raw_evidence: List[Dict[str, Any]]
    attachment_candidates: List[Dict[str, Any]]  # analog proxy sub_source="attachment_proxy"

    # ---------------------------------------------------------
    # Step 6b - parse_attachments
    # ---------------------------------------------------------
    attachment_docs: List[Dict[str, Any]]                # ParsedAttachment.to_dict() list
    attachment_native_candidates: List[Dict[str, Any]] # sub_source="attachment_native"
    attachment_extraction_metadata: List[Dict[str, Any]] # file_type, extraction_quality, etc.

    # ---------------------------------------------------------
    # Step 6c - promote_downstream_candidates (Phase 3)
    # ---------------------------------------------------------
    downstream_promoted_candidates: List[Dict[str, Any]]  # pattern-type downstream candidates

    # ---------------------------------------------------------
    # Step 7 - build_evidence
    # ---------------------------------------------------------
    candidate_evidence: List[Dict[str, Any]]  # CandidateEvidence dicts with all 7 sources

    # ---------------------------------------------------------
    # Step 8 - fuse_scores
    # ---------------------------------------------------------
    fused_candidates: List[Dict[str, Any]]
    fusion_profile: str                       # Which weight profile was applied

    # ---------------------------------------------------------
    # Step 9 - verify_candidates + finalize_selection
    # ---------------------------------------------------------
    verify_judgments: List[Dict[str, Any]]  # List[CandidateJudgment] dicts (Pass 1)
    selection_result: Dict[str, Any]        # SelectionResult-shaped dict (Pass 2)

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------
    directly_supported: List[Dict[str, Any]]
    pattern_inferred: List[Dict[str, Any]]
    no_evidence: List[Dict[str, Any]]
    selected_value_streams: List[Dict[str, Any]]  # compat union of direct + pattern
    rejected_candidates: List[Dict[str, Any]]

    # ---------------------------------------------------------
    # Taxonomy (Phase 1 foundation — consumed by later phases)
    # ---------------------------------------------------------
    taxonomy_registry: Optional[Dict[str, Any]]    # TaxonomyRegistry.model_dump()
    canonical_label_map: Optional[Dict[str, str]]  # alias (lower) -> canonical_name
    taxonomy_warnings: List[str]                   # non-fatal taxonomy issues

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------
    errors: List[str]
    warnings: List[str]
    timing: Dict[str, float]

    # ---------------------------------------------------------
    # Private pipeline config keys (injected by run_prediction_graph)
    # All _* keys use type: ignore at injection sites.
    # ---------------------------------------------------------
    _index_dir: str                   # FAISS summary index directory
    _ticket_chunks_dir: str           # ticket_chunks/ directory path
    _top_kg_candidates: int           # KG retrieval top_k
    _include_raw_evidence: bool       # whether to collect raw chunks
    _max_raw_evidence_tickets: int
    _min_candidate_floor: int         # minimum candidates for verifier
    _llm: Any                         # LLMService instance
    _theme_svc: Any                   # ThemeRetrievalService (overrides config)
    _intake_date: str                 # ISO-8601 date for theme leakage cutoff
    _attachment_contents: List[Dict[str, Any]]  # [{"filename": str, "content": bytes}]
    _services: Any                    # ServiceContainer (Phase 5)
    _taxonomy_registry: Any           # TaxonomyRegistry (Phase 1)
    _trace_prompt_callback: Any       # Optional callback for prompt tracing
    _trace_verify_prompt_callback: Any  # Optional callback for verify prompt tracing
