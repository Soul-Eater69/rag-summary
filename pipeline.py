"""
Summary-first RAG pipeline orchestrator (V5 architecture) — thin wrapper.

This module is the public API entry point. All orchestration logic lives in:
  - graph/build_prediction_graph.py  (LangGraph graph + sequential fallback)
  - graph/nodes.py                   (per-step node functions)
  - graph/edges.py                   (conditional routing logic)
  - chains/                          (LangChain prompt chains)
  - models/                          (Pydantic data contracts)
  - config/                          (YAML configs: capability_map, source_weights, function_vocab)

End-to-end flow (implemented in graph/nodes.py):
  1.  Clean + normalize new card text
  2.  Generate semantic summary (raw + canonical functions, capability tags)
  3.  Search FAISS summary index for top analog historical tickets
  4.  Collect VS evidence from analogs (including capability_tags, footprint)
  5.  Retrieve KG candidate value streams
  6.  Capability mapping enrichment
  7.  Extract card-level summary/chunk/footprint candidates
  8.  Collect raw chunks for top tickets (attachment proxy source)
  9.  Build CandidateEvidence objects from all 7 sources
  10. Source-aware fused ranking + candidate floor guardrail
  11. Two-pass LLM verifier (Pass 1: verify; Pass 2: finalize + calibrate)
  12. Finalize structured 3-class result
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from summary_rag.ingestion.faiss_indexer import DEFAULT_INDEX_DIR
from summary_rag.graph import run_prediction_graph


def run_summary_rag_pipeline(
    ppt_text: str,
    *,
    allowed_value_stream_names: Optional[List[str]] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    ticket_chunks_dir: str = "ticket_chunks",
    top_analogs: int = 5,
    top_kg_candidates: int = 20,
    include_raw_evidence: bool = True,
    max_raw_evidence_tickets: int = 3,
    min_candidate_floor: int = 8,
    debug_output_dir: Optional[str] = None,
    trace_mode: bool = False,
    llm=None,
) -> Dict[str, Any]:
    """
    Run the V5 summary-first RAG pipeline.

    Returns structured result with three-class output:
      - directly_supported: list of {entity_name, confidence, evidence}
      - pattern_inferred: list of {entity_name, confidence, evidence}
      - no_evidence: list of {entity_name, reason}

    Plus compatibility and diagnostic fields:
      - selected_value_streams: union of direct + pattern (compat)
      - rejected_candidates: list of {entity_name, reason}
      - new_card_summary: generated semantic summary dict
      - analog_tickets: FAISS-retrieved historical analogs
      - historical_value_stream_support: VS evidence from analogs
      - candidate_value_streams: fused CandidateEvidence objects
      - capability_mapping: capability map hits and promotions
      - raw_evidence: raw evidence snippets
      - warnings: list of non-fatal warning strings
      - timing: {"total_seconds": float}
    """
    result = run_prediction_graph(
        ppt_text,
        allowed_value_stream_names=allowed_value_stream_names,
        index_dir=index_dir,
        ticket_chunks_dir=ticket_chunks_dir,
        top_analogs=top_analogs,
        top_kg_candidates=top_kg_candidates,
        include_raw_evidence=include_raw_evidence,
        max_raw_evidence_tickets=max_raw_evidence_tickets,
        min_candidate_floor=min_candidate_floor,
        llm=llm,
    )

    if debug_output_dir:
        _persist_debug_artifacts(debug_output_dir, result)

    # Normalize return shape: ensure all expected keys are present
    return {
        "directly_supported": result.get("directly_supported", []),
        "pattern_inferred": result.get("pattern_inferred", []),
        "no_evidence": result.get("no_evidence", []),
        "selected_value_streams": result.get("selected_value_streams", []),
        "rejected_candidates": result.get("rejected_candidates", []),
        "new_card_summary": result.get("new_card_summary", {}),
        "analog_tickets": result.get("analog_tickets", []),
        "historical_value_stream_support": result.get("historical_value_stream_support", []),
        "candidate_value_streams": _format_candidate_vs(result.get("fused_candidates", [])),
        "capability_mapping": result.get("capability_mapping", {}),
        "raw_evidence": result.get("raw_evidence", []),
        "raw_response": result.get("selection_result", {}).get("raw_response"),
        "warnings": result.get("warnings", []),
        "timing": result.get("timing", {}),
        **({"trace": result.get("trace", {})} if trace_mode else {}),
    }


def _format_candidate_vs(fused_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shape fused candidates for the public candidate_value_streams field."""
    return [
        {
            "candidate_name": c.get("candidate_name", ""),
            "fused_score": c.get("fused_score", 0.0),
            "support_type": c.get("support_type", ""),
            "source_diversity_count": c.get("source_diversity_count", 0),
            "source_scores": c.get("source_scores", {}),
            "evidence_sources": c.get("evidence_sources", []),
        }
        for c in (fused_candidates or [])
    ]


def _persist_debug_artifacts(output_dir: str, result: Dict[str, Any]) -> None:
    """Write debug artifacts for a single pipeline run."""
    import json
    import os
    import logging

    log = logging.getLogger(__name__)
    try:
        os.makedirs(output_dir, exist_ok=True)
        artifacts = {
            "new_card_summary.json": result.get("new_card_summary", {}),
            "analog_tickets.json": result.get("analog_tickets", []),
            "vs_support.json": result.get("historical_value_stream_support", []),
            "kg_candidates.json": result.get("kg_candidates", []),
            "capability_mapping.json": result.get("capability_mapping", {}),
            "candidate_evidence.json": result.get("candidate_evidence", []),
            "fused_candidates.json": result.get("fused_candidates", []),
            "raw_evidence.json": result.get("raw_evidence", []),
            "selection_result.json": result.get("selection_result", {}),
            "eval_log.json": {
                "directly_supported": [vs.get("entity_name") for vs in result.get("directly_supported", [])],
                "pattern_inferred": [vs.get("entity_name") for vs in result.get("pattern_inferred", [])],
                "no_evidence": [vs.get("entity_name") for vs in result.get("no_evidence", [])],
                "warnings": result.get("warnings", []),
                "timing": result.get("timing", {}),
            },
        }
        for filename, data in artifacts.items():
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        log.info("[pipeline] Debug artifacts written to %s", output_dir)
    except Exception as exc:
        log.warning("[pipeline] Failed to write debug artifacts: %s", exc)
