"""
Summary-first RAG pipeline orchestrator (V5 architecture).

End-to-end flow:
1.  Clean + normalize new card text
2.  Generate semantic summary (raw + canonical functions, capability tags)
3.  Search FAISS summary index for top analog historical tickets
4.  Collect VS evidence from analogs (including capability_tags, footprint)
5.  Retrieve KG candidate value streams
6.  Capability mapping enrichment
7.  (Optional) Fetch raw chunks for top tickets  ← moved BEFORE evidence build
8.  Build CandidateEvidence objects from all sources
    (KG, historical, capability, attachment)
9.  Source-aware fused ranking + candidate floor guardrail
10. Two-pass LLM verifier (evidence verification + 3-class selection)
11. Finalize structured 3-class result
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from core.text import clean_ppt_text, normalize_for_search

from summary_rag.ingestion import generate_new_card_summary, build_retrieval_text
from summary_rag.ingestion.function_normalizer import normalize_functions
from summary_rag.ingestion.faiss_indexer import DEFAULT_INDEX_DIR
from summary_rag.retrieval import (
    retrieve_analog_tickets,
    retrieve_raw_evidence_for_tickets,
    retrieve_kg_candidates,
    collect_value_stream_evidence,
    collect_attachment_candidates,
)
from summary_rag.generation.capability_mapper import map_capabilities_to_candidates
from summary_rag.generation.candidate_evidence import build_candidate_evidence
from summary_rag.generation.fusion import compute_fused_scores, apply_candidate_floor
from summary_rag.generation.selector import select_value_streams
from summary_rag.generation.card_candidates import (
    extract_summary_candidates,
    extract_chunk_candidates,
    extract_historical_footprint_candidates,
)

logger = logging.getLogger(__name__)

_VSR_SUFFIX_RE = re.compile(r"\s*(\s*VSR[0-9A-Z-]*\s*)\s*$", re.IGNORECASE)
_EMPTY_PARENS_SUFFIX_RE = re.compile(r"\s*\(\s*\)\s*$")

def _clean_value_stream_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return ""
    cleaned = _VSR_SUFFIX_RE.sub("", cleaned)
    cleaned = _EMPTY_PARENS_SUFFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"\(-\)", "", cleaned).strip(" -")
    return cleaned

def _value_stream_key(name: str) -> str:
    return normalize_for_search(_clean_value_stream_name(name))


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
) -> Dict[str, Any]:
    """
    Run the V5 summary-first RAG pipeline.

    Returns structured result with three-class output:
      - directly_supported
      - pattern_inferred
      - no_evidence
    Plus compatibility fields (selected_value_streams, rejected_candidates).
    """
    t_start = time.time()
    warnings: List[str] = []
    trace: Dict[str, Any] = {}

    # -- Step 1: Clean and summarize new card --------------------------
    cleaned_text = clean_ppt_text(ppt_text)
    if trace_mode:
        trace["step1_input"] = {
            "label": "Clean & Summarize Input",
            "raw_text_length": len(ppt_text),
            "cleaned_text_length": len(cleaned_text),
            "cleaned_text_preview": cleaned_text[:200],
        }

    if not cleaned_text.strip():
        logger.warning("[SummaryRAG] Empty input after cleaning")
        return _empty_result("empty_input_text")

    t1 = time.time()
    try:
        new_card_summary = generate_new_card_summary(cleaned_text)
    except Exception as exc:
        logger.error("[SummaryRAG] Summary generation failed: %s", exc)
        warnings.append(f"summary_generation_failed: {type(exc).__name__}")
        new_card_summary = _deterministic_fallback_summary(cleaned_text)

    logger.info("[SummaryRAG] Step 1 (summary) done in %.2fs", time.time() - t1)
    if trace_mode:
        trace["step1_summary"] = {
            "label": "New Card Semantic Summary",
            "timing_s": round(time.time() - t1, 2),
            "fallback_used": "summary_generation_failed" in "".join(warnings),
            "summary": new_card_summary,
        }

    # -- Step 2: Search FAISS for analog tickets -----------------------
    t2 = time.time()
    analog_tickets: List[Dict[str, Any]] = []
    try:
        analog_tickets = retrieve_analog_tickets(
            new_card_summary,
            index_dir=index_dir,
            top_k=top_analogs,
        )
    except Exception as exc:
        logger.error("[SummaryRAG] FAISS search failed: %s", exc)
        warnings.append(f"faiss_search_failed: {type(exc).__name__}")

    logger.info("[SummaryRAG] Step 2 (FAISS search) done in %.2fs | %d analogs", time.time() - t2, len(analog_tickets))
    if trace_mode:
        trace["step2_faiss"] = {
            "label": "FAISS Analog Ticket Search",
            "timing_s": round(time.time() - t2, 2),
            "query_text": build_retrieval_text(new_card_summary),
            "top_k_requested": top_analogs,
            "results_count": len(analog_tickets),
            "analog_tickets": analog_tickets,
        }

    # -- Step 3: Collect VS evidence from analogs ----------------------
    t3 = time.time()
    vs_support = collect_value_stream_evidence(
        analog_tickets,
        ticket_chunks_dir=ticket_chunks_dir,
    )
    logger.info("[SummaryRAG] Step 3 (VS evidence) done in %.2fs | %d VS entries", time.time() - t3, len(vs_support))
    if trace_mode:
        trace["step3_vs_evidence"] = {
            "label": "Value Stream Evidence from Analogs",
            "timing_s": round(time.time() - t3, 2),
            "entries_count": len(vs_support),
            "vs_support": vs_support,
        }

    # -- Step 4: Retrieve KG candidates --------------------------------
    t4 = time.time()
    kg_query_text = build_retrieval_text(new_card_summary).strip() or cleaned_text
    kg_candidates: List[Dict[str, Any]] = []
    try:
        kg_candidates = retrieve_kg_candidates(
            kg_query_text,
            top_k=top_kg_candidates,
            allowed_names=allowed_value_stream_names,
        )
    except Exception as exc:
        logger.error("[SummaryRAG] KG retrieval failed: %s", exc)
        warnings.append(f"kg_retrieval_failed: {type(exc).__name__}")

    logger.info("[SummaryRAG] Step 4 (KG candidates) done in %.2fs | %d candidates", time.time() - t4, len(kg_candidates))
    if trace_mode:
        trace["step4_kg_candidates"] = {
            "label": "KG Candidate Value Streams",
            "timing_s": round(time.time() - t4, 2),
            "kg_query_text": kg_query_text,
            "top_k_requested": top_kg_candidates,
            "candidates_count": len(kg_candidates),
            "candidates": [
                {
                    "entity_id": c.get("entity_id", ""),
                    "entity_name": c.get("entity_name", ""),
                    "score": round(float(c.get("score") or c.get("best_score") or 0), 4),
                    "description": (c.get("description") or "")[:200],
                }
                for c in kg_candidates
            ],
        }

    # -- Step 5: Capability mapping ------------------------------------
    t5 = time.time()
    capability_mapping = map_capabilities_to_candidates(
        new_card_summary=new_card_summary,
        cleaned_text=cleaned_text,
        vs_support=vs_support,
        candidates=kg_candidates,
        allowed_value_stream_names=allowed_value_stream_names,
    )
    enriched_candidates = capability_mapping.get("enriched_candidates", kg_candidates)

    logger.info(
        "[SummaryRAG] Step 5 (capability mapping) done in %.2fs | hits=%d | promoted=%d",
        time.time() - t5,
        len(capability_mapping.get("capability_hits", [])),
        len(capability_mapping.get("promoted_value_streams", [])),
    )
    if trace_mode:
        trace["step5_capability_mapping"] = {
            "label": "Capability Mapping Candidate Enrichment",
            "timing_s": round(time.time() - t5, 2),
            "capability_hits": capability_mapping.get("capability_hits", []),
            "promoted_value_streams": capability_mapping.get("promoted_value_streams", []),
            "enriched_candidates_count": len(enriched_candidates),
        }

    # -- Step 5b: Card-level summary + chunk + footprint candidates ----------
    # These populate the summary, chunk, and additional historical source slots
    # in CandidateEvidence so they are non-zero when the card has explicit signals.
    allowed_set = (
        set(allowed_value_stream_names) if allowed_value_stream_names else None
    )
    summary_candidates = extract_summary_candidates(
        new_card_summary, allowed_names=allowed_set
    )
    chunk_candidates = extract_chunk_candidates(
        cleaned_text, allowed_names=allowed_set
    )
    # Historical footprint: infer additional pattern candidates from analog
    # capability_tags -- "what did similar tickets need downstream?"
    footprint_candidates = extract_historical_footprint_candidates(
        analog_tickets, allowed_names=allowed_set
    )
    logger.info(
        "[SummaryRAG] Step 5b (card candidates) | summary=%d | chunk=%d | footprint=%d",
        len(summary_candidates), len(chunk_candidates), len(footprint_candidates),
    )
    if trace_mode:
        trace["step5b_card_candidates"] = {
            "label": "Card-Level Summary / Chunk / Footprint Candidates",
            "summary_candidates": summary_candidates,
            "chunk_candidates": chunk_candidates,
            "footprint_candidates": footprint_candidates,
        }

    # -- Step 7: Optional raw evidence for top tickets (BEFORE evidence build) --
    # Must happen here so attachment snippets become a real CandidateEvidence source.
    raw_evidence: List[Dict[str, Any]] = []
    if include_raw_evidence and analog_tickets:
        t7 = time.time()
        top_ticket_ids = [
            t["ticket_id"] for t in analog_tickets[:max_raw_evidence_tickets]
            if t.get("ticket_id")
        ]
        try:
            raw_evidence = retrieve_raw_evidence_for_tickets(
                top_ticket_ids,
                ticket_chunks_dir=ticket_chunks_dir,
                query_text=kg_query_text,
            )
        except Exception as exc:
            logger.warning("[SummaryRAG] Raw evidence retrieval failed: %s", exc)

        logger.info("[SummaryRAG] Step 7 (raw evidence) done in %.2fs | %d snippets", time.time() - t7, len(raw_evidence))
        if trace_mode:
            trace["step7_raw_evidence"] = {
                "label": "Raw Evidence Snippets (attachment source)",
                "timing_s": round(time.time() - t7, 2),
                "tickets_queried": top_ticket_ids,
                "snippets_count": len(raw_evidence),
                "snippets": raw_evidence,
            }

    # -- Step 8: Build CandidateEvidence objects from all sources ----------
    t8 = time.time()

    # Historical candidates: include V5 capability/footprint from FAISS metadata
    historical_candidates = [
        {
            "entity_name": s.get("entity_name", ""),
            "score": float(s.get("best_score") or 0.0),
            # supporting_evidence carries functions + any explicit evidence snippets
            "supporting_evidence": (
                s.get("supporting_evidence", []) or s.get("supporting_functions", [])
            ),
            # V5 metadata for richer evidence context
            "capability_tags": s.get("capability_tags", []),
            "operational_footprint": s.get("operational_footprint", []),
        }
        for s in vs_support
    ]

    # KG candidates with normalized scores
    kg_for_evidence = [
        {
            "entity_name": c.get("entity_name", ""),
            "entity_id": c.get("entity_id", ""),
            "score": _normalize_kg_score(c),
            "description": (c.get("description") or "")[:200],
        }
        for c in kg_candidates
    ]

    # Attachment candidates derived from raw evidence snippets
    attachment_candidates = collect_attachment_candidates(raw_evidence, analog_tickets)

    # footprint_candidates contribute to the historical source alongside
    # vs_support-derived historical_candidates
    all_historical = historical_candidates + footprint_candidates

    candidate_evidence = build_candidate_evidence(
        kg_candidates=kg_for_evidence,
        historical_candidates=all_historical,
        capability_candidates=capability_mapping.get("capability_candidates", []),
        chunk_candidates=chunk_candidates,
        attachment_candidates=attachment_candidates,
        # summary_candidates go into summary source slot
    )
    # Inject summary candidates separately so they land in SOURCE_SUMMARY
    from summary_rag.generation.candidate_evidence import SOURCE_SUMMARY
    _inject_summary_candidates(candidate_evidence, summary_candidates)

    logger.info(
        "[SummaryRAG] Step 8 (CandidateEvidence) done in %.2fs | %d candidates | attachment_source=%d",
        time.time() - t8, len(candidate_evidence), len(attachment_candidates),
    )
    if trace_mode:
        trace["step8_candidate_evidence"] = {
            "label": "CandidateEvidence Objects (all sources)",
            "timing_s": round(time.time() - t8, 2),
            "candidates_count": len(candidate_evidence),
            "attachment_candidates_count": len(attachment_candidates),
            "candidates_preview": [
                {
                    "name": c.get("candidate_name"),
                    "support_type": c.get("support_type"),
                    "diversity": c.get("source_diversity_count"),
                    "sources": c.get("evidence_sources"),
                }
                for c in candidate_evidence[:15]
            ],
        }

    # -- Step 9: Source-aware fused ranking + candidate floor -----------
    t9a = time.time()
    fused_candidates = compute_fused_scores(candidate_evidence)
    fused_candidates = apply_candidate_floor(fused_candidates, min_candidates=min_candidate_floor)

    logger.info("[SummaryRAG] Step 9a (fusion) done in %.2fs | %d candidates after floor", time.time() - t9a, len(fused_candidates))
    if trace_mode:
        trace["step9a_fusion"] = {
            "label": "Source-Aware Fused Ranking",
            "timing_s": round(time.time() - t9a, 2),
            "candidates_count": len(fused_candidates),
            "top_candidates": [
                {
                    "name": c.get("candidate_name"),
                    "fused_score": c.get("fused_score"),
                    "support_type": c.get("support_type"),
                    "source_scores": c.get("source_scores"),
                }
                for c in fused_candidates[:10]
            ],
        }

    # -- Step 9b: Two-pass LLM verifier ---------------------------------
    t9b = time.time()
    selection_result = select_value_streams(
        new_card_summary=new_card_summary,
        analog_tickets=analog_tickets,
        candidates=fused_candidates,
        raw_evidence=raw_evidence,
        vs_support=vs_support,
        allowed_value_stream_names=allowed_value_stream_names,
    )

    logger.info(
        "[SummaryRAG] Step 9b (LLM verifier) done in %.2fs | direct=%d | pattern=%d | no_evidence=%d",
        time.time() - t9b,
        len(selection_result.get("directly_supported", [])),
        len(selection_result.get("pattern_inferred", [])),
        len(selection_result.get("no_evidence", [])),
    )
    if trace_mode:
        trace["step9b_llm_selection"] = {
            "label": "Two-Pass LLM Verifier + Selection",
            "timing_s": round(time.time() - t9b, 2),
            "system_prompt": selection_result.get("prompt_system", ""),
            "user_prompt": selection_result.get("prompt_user", ""),
            "candidates_sent_to_llm": len(fused_candidates),
            "raw_llm_response": selection_result.get("raw_response", ""),
            "directly_supported_count": len(selection_result.get("directly_supported", [])),
            "pattern_inferred_count": len(selection_result.get("pattern_inferred", [])),
            "no_evidence_count": len(selection_result.get("no_evidence", [])),
        }

    # -- Step 10: Finalize 3-class output -----------------------------
    # Build final_selected from the 3-class output directly (not from the
    # compat selected_value_streams field) to ensure the 3-class contract
    # is the authoritative source of truth.
    seen_names: set[str] = set()

    def _dedup_vs_list(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for vs in items:
            cleaned = _clean_value_stream_name(vs.get("entity_name") or "")
            if not cleaned:
                continue
            key = _value_stream_key(cleaned)
            if not key or key in seen_names:
                continue
            seen_names.add(key)
            out.append({**vs, "entity_name": cleaned})
        return out

    directly_supported = _dedup_vs_list(selection_result.get("directly_supported", []))
    pattern_inferred = _dedup_vs_list(selection_result.get("pattern_inferred", []))
    no_evidence = selection_result.get("no_evidence", [])

    # Flat union (direct first, then pattern) for compat consumers
    final_selected = directly_supported + pattern_inferred

    elapsed = time.time() - t_start
    logger.info(
        "[SummaryRAG] DONE in %.2fs | direct=%d | pattern=%d | no_evidence=%d | analogs=%d | warnings=%s",
        elapsed,
        len(directly_supported),
        len(pattern_inferred),
        len(no_evidence),
        len(analog_tickets), warnings or "none",
    )

    # Evaluation log
    eval_log: Dict[str, Any] = {
        "top_analogs": [
            {"ticket_id": t.get("ticket_id"), "score": t.get("score"), "rank": t.get("rank")}
            for t in analog_tickets
        ],
        "kg_candidates": [c.get("entity_name") for c in kg_candidates],
        "capability_hits": capability_mapping.get("capability_hits", []),
        "promoted_value_streams": capability_mapping.get("promoted_value_streams", []),
        "candidate_evidence_count": len(candidate_evidence),
        "fused_candidates_count": len(fused_candidates),
        "directly_supported": [vs.get("entity_name") for vs in directly_supported],
        "pattern_inferred": [vs.get("entity_name") for vs in pattern_inferred],
        "no_evidence": [vs.get("entity_name") for vs in no_evidence],
        "selected_value_streams": [vs.get("entity_name") for vs in final_selected],
        "attachment_candidates_count": len(attachment_candidates),
        "summary_candidates_count": len(summary_candidates),
        "chunk_candidates_count": len(chunk_candidates),
        "footprint_candidates_count": len(footprint_candidates),
        "raw_evidence_count": len(raw_evidence),
        "warnings": warnings,
        "timing_seconds": round(elapsed, 2),
    }

    logger.info("[SummaryRAG] eval log %s", json.dumps(eval_log, default=str))

    if debug_output_dir:
        _persist_debug_artifacts(
            debug_output_dir,
            new_card_summary=new_card_summary,
            analog_tickets=analog_tickets,
            vs_support=vs_support,
            kg_candidates=kg_candidates,
            capability_mapping=capability_mapping,
            candidate_evidence=candidate_evidence,
            fused_candidates=fused_candidates,
            raw_evidence=raw_evidence,
            selection_result=selection_result,
            eval_log=eval_log,
        )

    return {
        # V5 three-class output (authoritative)
        "directly_supported": directly_supported,
        "pattern_inferred": pattern_inferred,
        "no_evidence": no_evidence,

        # Compat: flat selected list (direct + pattern, deduplicated)
        "selected_value_streams": final_selected,
        "rejected_candidates": selection_result.get("rejected_candidates", []),

        # Intermediate artifacts
        "new_card_summary": new_card_summary,
        "analog_tickets": analog_tickets,
        "historical_value_stream_support": vs_support,
        "candidate_value_streams": [
            {
                "candidate_name": c.get("candidate_name", ""),
                "fused_score": c.get("fused_score", 0.0),
                "support_type": c.get("support_type", ""),
                "source_diversity_count": c.get("source_diversity_count", 0),
                "source_scores": c.get("source_scores", {}),
                "evidence_sources": c.get("evidence_sources", []),
            }
            for c in fused_candidates
        ],
        "capability_mapping": capability_mapping,
        "raw_evidence": raw_evidence,
        "raw_response": selection_result.get("raw_response"),
        "warnings": warnings,
        "timing": {
            "total_seconds": round(elapsed, 2),
        },
        **(
            {"trace": trace,
             "prompt_system": selection_result.get("prompt_system", ""),
             "prompt_user": selection_result.get("prompt_user", "")}
            if trace_mode else {}
        ),
    }


def _inject_summary_candidates(
    candidate_evidence: List[Dict[str, Any]],
    summary_candidates: List[Dict[str, Any]],
) -> None:
    """
    Merge summary-source candidates into existing CandidateEvidence objects.

    For candidates already present (by name), update their summary source score.
    For new names, append a new CandidateEvidence entry with only summary score set.
    """
    from summary_rag.generation.candidate_evidence import (
        SOURCE_SUMMARY, ALL_SOURCES, _classify_support_type
    )
    from summary_rag.ingestion.adapters import normalize_text as _norm

    by_name = {_norm(c.get("candidate_name", "")): c for c in candidate_evidence}

    for sc in summary_candidates:
        name = (sc.get("entity_name") or "").strip()
        key = _norm(name)
        if not name or not key:
            continue
        score = float(sc.get("score") or 0.0)

        if key in by_name:
            entry = by_name[key]
            entry["source_scores"][SOURCE_SUMMARY] = max(
                entry["source_scores"].get(SOURCE_SUMMARY, 0.0), score
            )
            if SOURCE_SUMMARY not in entry.get("evidence_sources", []):
                entry.setdefault("evidence_sources", []).append(SOURCE_SUMMARY)
        else:
            new_entry: Dict[str, Any] = {
                "candidate_id": "",
                "candidate_name": name,
                "source_scores": {s: 0.0 for s in ALL_SOURCES},
                "evidence_sources": [SOURCE_SUMMARY],
                "evidence_snippets": [],
                "fused_score": 0.0,
                "support_confidence": 0.0,
                "source_diversity_count": 1,
                "support_type": "direct",
                "contradictions": [],
            }
            new_entry["source_scores"][SOURCE_SUMMARY] = score
            candidate_evidence.append(new_entry)
            by_name[key] = new_entry

    # Recompute support_type for any updated entries
    for entry in candidate_evidence:
        entry["support_type"] = _classify_support_type(entry)


def _normalize_kg_score(candidate: Dict[str, Any]) -> float:
    """Normalize a KG candidate score to 0..1 range."""
    raw = float(candidate.get("score") or candidate.get("best_score") or 0.0)
    if raw <= 1.0:
        return max(0.0, raw)
    # Large reranker scores: use sigmoid-like normalization
    return min(1.0, raw / (raw + 10.0))


def _persist_debug_artifacts(
    output_dir: str,
    *,
    new_card_summary: Dict[str, Any],
    analog_tickets: List[Dict[str, Any]],
    vs_support: List[Dict[str, Any]],
    kg_candidates: List[Dict[str, Any]],
    capability_mapping: Dict[str, Any],
    candidate_evidence: List[Dict[str, Any]],
    fused_candidates: List[Dict[str, Any]],
    raw_evidence: List[Dict[str, Any]],
    selection_result: Dict[str, Any],
    eval_log: Dict[str, Any],
) -> None:
    """Write debug artifacts for a single pipeline run."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        artifacts = {
            "new_card_summary.json": new_card_summary,
            "analog_tickets.json": analog_tickets,
            "vs_support.json": vs_support,
            "kg_candidates.json": kg_candidates,
            "capability_mapping.json": capability_mapping,
            "candidate_evidence.json": candidate_evidence,
            "fused_candidates.json": fused_candidates,
            "raw_evidence.json": raw_evidence,
            "selection_result.json": {
                "directly_supported": selection_result.get("directly_supported", []),
                "pattern_inferred": selection_result.get("pattern_inferred", []),
                "no_evidence": selection_result.get("no_evidence", []),
                "raw_response": selection_result.get("raw_response"),
            },
            "eval_log.json": eval_log,
        }
        for filename, data in artifacts.items():
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info("[SummaryRAG] Debug artifacts written to %s", output_dir)
    except Exception as exc:
        logger.warning("[SummaryRAG] Failed to write debug artifacts: %s", exc)


ACTOR_KEYWORDS = [
    "member", "provider", "broker", "employer", "plan sponsor",
    "internal ops", "clinical", "care manager", "agent", "beneficiary",
]

FUNCTION_KEYWORDS = [
    "claims", "enrollment", "billing", "prior auth", "authorization",
    "eligibility", "referral", "appeals", "grievance", "pharmacy",
    "care management", "utilization", "network", "credentialing",
    "payment", "risk adjustment", "quality", "reporting",
]

DOMAIN_KEYWORDS = {
    "clinical": ["clinical", "care", "health", "medical", "diagnosis", "treatment"],
    "financial": ["billing", "claims", "payment", "cost", "revenue", "finance"],
    "operational": ["process", "workflow", "efficiency", "ops", "operational"],
    "it": ["system", "platform", "integration", "api", "migration", "data", "digital"],
    "regulatory": ["regulatory", "compliance", "hipaa", "cms", "mandate", "audit"],
}

def _deterministic_fallback_summary(cleaned_text: str) -> Dict[str, Any]:
    """
    Build a best-effort summary using keyword heuristics when LLM fails.

    V5: populates both raw and canonical function fields.
    """
    lower = cleaned_text.lower()
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]

    first_line = lines[0] if lines else ""
    short_summary = first_line if len(first_line) > 20 else cleaned_text[:200]

    actors = [kw.title() for kw in ACTOR_KEYWORDS if kw in lower]
    raw_functions = [kw.title() for kw in FUNCTION_KEYWORDS if kw in lower]
    canonical_functions = normalize_functions(raw_functions)

    domain_tags = []
    for domain, signals in DOMAIN_KEYWORDS.items():
        if any(s in lower for s in signals):
            domain_tags.append(domain.title())

    return {
        "short_summary": short_summary,
        "business_goal": lines[1] if len(lines) > 1 else "",
        "actors": actors,
        "direct_functions_raw": raw_functions,
        "direct_functions_canonical": canonical_functions,
        "implied_functions_raw": [],
        "implied_functions_canonical": [],
        "direct_functions": canonical_functions,
        "implied_functions": [],
        "change_types": [],
        "domain_tags": domain_tags,
        "capability_tags": [],
        "operational_footprint": [],
        "evidence_sentences": lines[:3],
    }

def _empty_result(error: str) -> Dict[str, Any]:
    return {
        "directly_supported": [],
        "pattern_inferred": [],
        "no_evidence": [],
        "selected_value_streams": [],
        "rejected_candidates": [],
        "new_card_summary": {},
        "analog_tickets": [],
        "historical_value_stream_support": [],
        "candidate_value_streams": [],
        "raw_evidence": [],
        "raw_response": None,
        "warnings": [error],
        "error": error,
        "timing": {},
    }
