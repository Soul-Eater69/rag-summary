"""
Summary-first RAG pipeline orchestrator (V6 architecture) - thin wrapper.

This module is the public API entry point. All orchestration logic lives in:
  - graph/build_prediction_graph.py  (LangGraph 16-node graph + sequential fallback)
  - graph/nodes.py                   (per-step node functions)
  - graph/edges.py                   (conditional routing logic)
  - chains/                          (LangChain prompt chains)
  - models/                          (Pydantic data contracts)
  - config/                          (YAML configs: capability_map, source_weights,
                                      function_vocab, stream_signal_rules,
                                      downstream_promotion_rules, taxonomy_policy_rules,
                                      historical_label_priors, taxonomy_registry)

End-to-end flow - 16 nodes (implemented in graph/nodes.py):
  1.  clean_and_summarize        - normalize text, generate semantic summary
  2.  retrieve_analogs           - FAISS search for top analog historical tickets
  3.  collect_vs_evidence        - VS support from analogs + bundle/downstream pattern detection
  4.  retrieve_kg                - KG candidate value streams
  5.  retrieve_themes            - theme-cluster candidates (auto-discovers FAISS index;
                                   falls back to always-active KeywordThemeService)
  6.  map_capabilities           - capability map enrichment
  7.  extract_card_candidates    - summary/chunk/attachment_heuristic/footprint candidates
  8.  collect_raw_evidence       - raw evidence chunks + attachment proxy candidates
  9.  parse_attachments          - section-level parsing + attachment native candidates
                                   (reads _attachment_contents or falls back to card text)
  10. promote_downstream_candidates - Phase 3: pattern-based downstream candidate promotion
  11. build_evidence             - merge all 7 sources into CandidateEvidence; inject
                                   bundle/downstream pattern score boosts
  12. fuse_scores                - source-aware weighted fusion + quality multipliers +
                                   theme promotion bonus + candidate floor guardrail
  13. verify_candidates          - Pass 1 LLM verifier + per-candidate judgments
  14. taxonomy_policy_rerank     - Phase 4: label eligibility reranking via policy rules,
                                   sibling dominance, and historical priors
  15. finalize_selection         - Pass 2 LLM selector consuming reranked candidates
  16. finalize_output            - dedup, normalize, produce final output

7 evidence sources: chunk | summary | attachment | theme | kg | historical | capability
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime, timezone

from rag_summary.ingestion.faiss_indexer import DEFAULT_INDEX_DIR
from rag_summary.graph import run_prediction_graph


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
    theme_svc=None,
    intake_date: Optional[str] = None,
    services=None,
    taxonomy_registry=None,
) -> Dict[str, Any]:
    """
    Run the V6 summary-first RAG pipeline.

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
    effective_debug_dir = debug_output_dir
    if trace_mode and not effective_debug_dir:
        effective_debug_dir = "rag_summary/outputs"

    live_run_number: Optional[int] = None
    live_payload: Optional[Dict[str, Any]] = None

    def _write_live_snapshot() -> None:
        if not effective_debug_dir or live_payload is None:
            return
        os.makedirs(effective_debug_dir, exist_ok=True)
        current_path = os.path.join(effective_debug_dir, "output1_current.json")
        with open(current_path, "w", encoding="utf-8") as f:
            json.dump(live_payload, f, ensure_ascii=False, indent=2, default=str)
        if live_run_number is not None:
            run_path = os.path.join(effective_debug_dir, f"output1_run_{live_run_number:04d}.json")
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(live_payload, f, ensure_ascii=False, indent=2, default=str)

    if effective_debug_dir:
        os.makedirs(effective_debug_dir, exist_ok=True)
        live_run_number = _next_run_number(effective_debug_dir)
        live_payload = {
            "run_number": live_run_number,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "latest_node": None,
            "node_outputs": {},
            "verify_prompt": {},
            "final_prompt": {},
            "final_response": {},
            "warnings": [],
            "timing": {},
        }
        _write_live_snapshot()

    def _on_node_update(node_name: str, updates: Dict[str, Any], merged_state: Dict[str, Any]) -> None:
        if live_payload is None:
            return
        live_payload["latest_node"] = node_name
        live_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        live_payload.setdefault("node_outputs", {})[node_name] = updates
        prompt = merged_state.get("final_prompt")
        if isinstance(prompt, dict) and prompt:
            live_payload["final_prompt"] = prompt
        _write_live_snapshot()

    def _on_final_prompt(prompt_payload: Dict[str, str]) -> None:
        if live_payload is None:
            return
        live_payload["final_prompt"] = prompt_payload or {}
        live_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_live_snapshot()

    def _on_verify_prompt(prompt_payload: Dict[str, str]) -> None:
        if live_payload is None:
            return
        live_payload["verify_prompt"] = prompt_payload or {}
        live_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_live_snapshot()

    try:
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
            theme_svc=theme_svc,
            intake_date=intake_date,
            services=services,
            taxonomy_registry=taxonomy_registry,
            trace_mode=trace_mode,
            trace_node_callback=_on_node_update if effective_debug_dir else None,
            trace_prompt_callback=_on_final_prompt if effective_debug_dir else None,
            trace_verify_prompt_callback=_on_verify_prompt if effective_debug_dir else None,
        )
    except Exception as exc:
        if live_payload is not None:
            live_payload["status"] = "failed"
            live_payload["error"] = f"{type(exc).__name__}: {exc}"
            live_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_live_snapshot()
        raise

    if effective_debug_dir:
        _persist_debug_artifacts(effective_debug_dir, result, run_number=live_run_number)

    if live_payload is not None:
        live_payload["status"] = "completed"
        live_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        live_payload["final_response"] = {
            "directly_supported": result.get("directly_supported", []),
            "pattern_inferred": result.get("pattern_inferred", []),
            "no_evidence": result.get("no_evidence", []),
            "selected_value_streams": result.get("selected_value_streams", []),
            "rejected_candidates": result.get("rejected_candidates", []),
        }
        live_payload["warnings"] = result.get("warnings", [])
        live_payload["timing"] = result.get("timing", {})
        if not live_payload.get("final_prompt"):
            prompt = result.get("final_prompt") or (result.get("trace") or {}).get("final_prompt", {})
            if isinstance(prompt, dict):
                live_payload["final_prompt"] = prompt
        _write_live_snapshot()

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
        # V6 diagnostic fields
        "theme_source_status": result.get("theme_source_status", {}),
        "fusion_profile": result.get("fusion_profile", "default"),
        "final_prompt": result.get("final_prompt", {}),
        # Phase 4: taxonomy policy reranker outputs
        "taxonomy_reranked_candidates": result.get("taxonomy_reranked_candidates", []),
        "taxonomy_suppressed_candidates": result.get("taxonomy_suppressed_candidates", []),
        "taxonomy_decisions": result.get("taxonomy_decisions", []),
        "downstream_promoted_candidates": result.get("downstream_promoted_candidates", []),
        "bundle_patterns": result.get("bundle_patterns", []),
        "downstream_chains": result.get("downstream_chains", []),
        # Taxonomy registry diagnostics
        "canonical_label_map": result.get("canonical_label_map", {}),
        "taxonomy_warnings": result.get("taxonomy_warnings", []),
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


def _next_run_number(output_dir: str) -> int:
    output1_path = os.path.join(output_dir, "output1.json")
    if not os.path.exists(output1_path):
        return 1
    try:
        with open(output1_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        runs = history.get("runs", []) if isinstance(history, dict) else []
        if isinstance(runs, list) and runs:
            last = runs[-1]
            if isinstance(last, dict):
                return int(last.get("run_number", 0) or 0) + 1
    except Exception:
        return 1
    return 1


def _persist_debug_artifacts(
    output_dir: str,
    result: Dict[str, Any],
    *,
    run_number: Optional[int] = None,
) -> None:
    """Write debug artifacts for a single pipeline run."""
    import json
    import os
    import logging
    from datetime import datetime, timezone

    log = logging.getLogger(__name__)
    try:
        os.makedirs(output_dir, exist_ok=True)
        log.info("[_persist_debug_artifacts] Created directory: %s", output_dir)
        trace = result.get("trace") or {}

        # Extract final prompt from node outputs if not directly in result
        final_prompt = result.get("final_prompt") or {}
        if not final_prompt and trace.get("node_outputs"):
            # Try to get from finalize_selection node
            finalize_node = trace.get("node_outputs", {}).get("finalize_selection", {})
            if isinstance(finalize_node, dict):
                final_prompt = finalize_node.get("final_prompt", {})

        run_payload = {
            "node_outputs": trace.get("node_outputs", {}),
            "final_prompt": final_prompt,
            "final_response": {
                "directly_supported": result.get("directly_supported", []),
                "pattern_inferred": result.get("pattern_inferred", []),
                "no_evidence": result.get("no_evidence", []),
                "selected_value_streams": result.get("selected_value_streams", []),
                "rejected_candidates": result.get("rejected_candidates", []),
            },
            "timing": result.get("timing", {}),
            "warnings": result.get("warnings", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        output1_path = os.path.join(output_dir, "output1.json")
        history: Dict[str, Any] = {}
        if os.path.exists(output1_path):
            try:
                with open(output1_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                    history = existing
            except Exception:
                history = {"runs": []}

        runs = history.get("runs", [])
        if not isinstance(runs, list):
            runs = []
        computed_run_number = run_number if run_number is not None else _next_run_number(output_dir)
        run_payload["run_number"] = computed_run_number
        runs = [r for r in runs if not (isinstance(r, dict) and int(r.get("run_number", 0) or 0) == computed_run_number)]
        runs.append(run_payload)
        runs.sort(key=lambda r: int(r.get("run_number", 0) or 0) if isinstance(r, dict) else 0)
        history["runs"] = runs
        history["latest_run_number"] = computed_run_number
        history["run_count"] = len(runs)

        canonical_prediction_labels: List[str] = []
        if result.get("selected_value_streams"):
            canonical_prediction_labels = sorted({
                (c.get("entity_name") or "").strip()
                for c in result.get("selected_value_streams", [])
                if (c.get("entity_name") or "").strip()
            })

        canonical_gt_labels = result.get("canonical_gt_labels", [])
        stream_signal_hits = result.get("stream_signal_hits", [])

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
            # Phase 3: downstream promotion artifacts
            "downstream_promoted_candidates.json": result.get("downstream_promoted_candidates", []),
            # V6: historical footprint pattern artifacts
            "bundle_patterns.json": result.get("bundle_patterns", []),
            "downstream_chains.json": result.get("downstream_chains", []),
            "theme_candidates.json": result.get("theme_candidates", []),
            # V6: attachment parsing artifacts
            "attachment_docs.json": result.get("attachment_docs", []),
            "attachment_native_candidates.json": result.get("attachment_native_candidates", []),
            "attachment_extraction_metadata.json": result.get("attachment_extraction_metadata", []),
            # V6: theme and fusion diagnostics
            "theme_source_status.json": result.get("theme_source_status", {}),
            "theme_debug.json": result.get("theme_debug", {}),
            "fusion_profile.json": {"profile": result.get("fusion_profile", "default")},
            # Phase 4: taxonomy policy reranker decision artifacts
            "taxonomy_reranked_candidates.json": result.get("taxonomy_reranked_candidates", []),
            "taxonomy_suppressed_candidates.json": result.get("taxonomy_suppressed_candidates", []),
            "taxonomy_decisions.json": result.get("taxonomy_decisions", []),
            "canonical_prediction_labels.json": canonical_prediction_labels,
            "canonical_gt_labels.json": canonical_gt_labels,
            "stream_signal_hits.json": stream_signal_hits,
            # Taxonomy registry diagnostics
            "canonical_label_map.json": result.get("canonical_label_map", {}),
            "taxonomy_warnings.json": result.get("taxonomy_warnings", []),
            "eval_log.json": {
                "directly_supported": [vs.get("entity_name") for vs in result.get("directly_supported", [])],
                "pattern_inferred": [vs.get("entity_name") for vs in result.get("pattern_inferred", [])],
                "no_evidence": [vs.get("entity_name") for vs in result.get("no_evidence", [])],
                "warnings": result.get("warnings", []),
                "timing": result.get("timing", {}),
            },
        }

        for filename, data in artifacts.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            log.debug("[_persist_debug_artifacts] Wrote %s", filename)

        with open(output1_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)
        log.info("[_persist_debug_artifacts] Wrote output1.json (run %d)", computed_run_number)

        per_run_path = os.path.join(output_dir, f"output1_run_{computed_run_number:04d}.json")
        with open(per_run_path, "w", encoding="utf-8") as f:
            json.dump(run_payload, f, ensure_ascii=False, indent=2, default=str)
        log.info("[_persist_debug_artifacts] Wrote per-run file %s", f"output1_run_{computed_run_number:04d}.json")

        log.info("[_persist_debug_artifacts] All artifacts written to %s", output_dir)
    except Exception as exc:
        log.error("[_persist_debug_artifacts] Failed to write debug artifacts: %s", exc, exc_info=True)
