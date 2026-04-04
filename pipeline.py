"""
Summary-first RAG pipeline orchestrator.

End-to-end flow:
1. Clean + normalize new card text
2. Generate semantic summary for new card
3. Search FAISS summary index for top analog historical tickets
4. Collect VS evidence from analog tickets
5. Retrieve KG candidate value streams
6. Capability mapping candidate enrichment
7. Optional: fetch raw chunks for top shortlisted tickets
8. Build compact evidence package + LLM selection
9. Inject canonical defaults
10. Return structured result
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from core.text import clean_ppt_text, normalize_for_search

from .ingestion import generate_new_card_summary, build_retrieval_text
from .ingestion_faiss_indexer import DEFAULT_INDEX_DIR
from .retrieval import (
    retrieve_analog_tickets,
    retrieve_raw_evidence_for_tickets,
    retrieve_kg_candidates,
)
from .retrieval_summary_retriever import collect_value_stream_evidence
from .generation import select_value_streams, map_capabilities_to_candidates

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

def _candidate_raw_score(candidate: Dict[str, Any]) -> float:
    """Return the best available raw score from a candidate payload."""
    return float(candidate.get("score") or candidate.get("best_score") or candidate.get("_aggregated_best_score") or 0.0)

def _normalize_candidate_scores(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize heterogeneous candidate score scales to 0..1.

    KG retrieval may return large reranker scores (for example hundreds), while
    capability and historical support paths often use 0..1 like scores. This
    function makes scores comparable for ranking, display, and LLM prompting.
    """
    if not candidates:
        return []

    normalized = [dict(candidate) for candidate in candidates]
    raw_scores = [_candidate_raw_score(candidate) for candidate in normalized]
    max_score = max(raw_scores) if raw_scores else 0.0
    min_score = min(raw_scores) if raw_scores else 0.0
    span = max_score - min_score

    for candidate, raw_score in zip(normalized, raw_scores):
        candidate["raw_score"] = round(raw_score, 4)
        if max_score <= 1.0:
            score = max(0.0, min(1.0, raw_score))
        elif span <= 1e-9:
            score = 1.0 if raw_score > 0 else 0.0
        else:
            score = (raw_score - min_score) / span
        candidate["score"] = round(score, 4)

    normalized.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return normalized

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
    debug_output_dir: Optional[str] = None,
    trace_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run the summary-first RAG pipeline.

    Flow:
      1. Clean text + generate new-card summary
      2. Search FAISS for analog historical tickets
      3. Collect VS evidence from analogs
      4. Retrieve KG candidates
          5. Capability mapping candidate enrichment
          6. Score normalization across candidate sources
          7. Optional raw chunk lookup for top tickets
          8. LLM selection with compact evidence
          9. De-dup and finalize selected streams

    Returns structured result compatible with the existing API response shape.
    """
    t_start = time.time()
    warnings: List[str] = []
    trace: Dict[str, Any] = {}

    # - Step 1: Clean and summarize new card ------------------------
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

    # - Step 2: Search FAISS for analog tickets ----------------------
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

    # - Step 3: Collect VS evidence from analogs --------------------
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

    # - Step 4: Retrieve KG candidates -------------------------------
    # Use the summary retrieval text so KG search is fully summary-first.
    # Fall back to cleaned_text if the summary is empty (e.g. fallback path).
    t4 = time.time()
    kg_query_text = build_retrieval_text(new_card_summary).strip() or cleaned_text
    candidates: List[Dict[str, Any]] = []

    try:
        candidates = retrieve_kg_candidates(
            kg_query_text,
            top_k=top_kg_candidates,
            allowed_names=allowed_value_stream_names,
        )
    except Exception as exc:
        logger.error("[SummaryRAG] KG retrieval failed: %s", exc)
        warnings.append(f"kg_retrieval_failed: {type(exc).__name__}")

    logger.info("[SummaryRAG] Step 4 (KG candidates) done in %.2fs | %d candidates", time.time() - t4, len(candidates))
    if trace_mode:
        trace["step4_kg_candidates"] = {
            "label": "KG Candidate Value Streams",
            "timing_s": round(time.time() - t4, 2),
            "kg_query_text": kg_query_text,
            "top_k_requested": top_kg_candidates,
            "candidates_count": len(candidates),
            "candidates": [
                {
                    "entity_id": c.get("entity_id", ""), "entity_name": c.get("entity_name", ""),
                    "score": round(float(c.get("score") or c.get("best_score") or 0), 4),
                    "description": (c.get("description") or "")[:200]}
                for c in candidates
            ]
        }

    # - Step 5: Capability mapping candidate enrichment --------------
    t5 = time.time()
    capability_mapping = map_capabilities_to_candidates(
        new_card_summary=new_card_summary,
        cleaned_text=cleaned_text,
        vs_support=vs_support,
        candidates=candidates,
        allowed_value_stream_names=allowed_value_stream_names,
    )
    enriched_candidates = capability_mapping.get("enriched_candidates", candidates)
    enriched_candidates = _normalize_candidate_scores(enriched_candidates)

    logger.info(
        "[SummaryRAG] Step 5 (capability mapping) done in %.2fs | hits=%d | promoted=%d | candidates=%d",
        time.time() - t5,
        len(capability_mapping.get("capability_hits", [])),
        len(capability_mapping.get("promoted_value_streams", [])),
        len(enriched_candidates),
    )
    if trace_mode:
        trace["step5_capability_mapping"] = {
            "label": "Capability Mapping Candidate Enrichment",
            "timing_s": round(time.time() - t5, 2),
            "capability_hits": capability_mapping.get("capability_hits", []),
            "promoted_value_streams": capability_mapping.get("promoted_value_streams", []),
            "enriched_candidates_count": len(enriched_candidates),
            "enriched_candidates_preview": [
                {
                    "entity_name": c.get("entity_name"),
                    "score": round(float(c.get("score") or c.get("best_score") or 0.0), 4),
                    "source": c.get("source"),
                    "promotion_reason": c.get("promotion_reason"),
                }
                for c in enriched_candidates[:25]
            ]
        }

    # - Step 6: Optional raw evidence for top tickets ----------------
    raw_evidence: List[Dict[str, Any]] = []
    if include_raw_evidence and analog_tickets:
        t6 = time.time()
        top_ticket_ids = [t["ticket_id"] for t in analog_tickets[:max_raw_evidence_tickets] if t.get("ticket_id")]
        try:
            raw_evidence = retrieve_raw_evidence_for_tickets(
                top_ticket_ids,
                ticket_chunks_dir=ticket_chunks_dir,
                query_text=kg_query_text,
            )
        except Exception as exc:
            logger.warning("[SummaryRAG] Raw evidence retrieval failed: %s", exc)

        logger.info("[SummaryRAG] Step 6 (raw evidence) done in %.2fs | %d snippets", time.time() - t6, len(raw_evidence))
        if trace_mode:
            trace["step6_raw_evidence"] = {
                "label": "Raw Evidence Snippets (Top Ticket Chunks)",
                "timing_s": round(time.time() - t6, 2),
                "tickets_queried": top_ticket_ids,
                "snippets_count": len(raw_evidence),
                "snippets": raw_evidence,
            }

    # - Step 7: LLM selection ----------------------------------------
    t7 = time.time()
    selection_result = select_value_streams(
        new_card_summary=new_card_summary,
        analog_tickets=analog_tickets,
        candidates=enriched_candidates,
        raw_evidence=raw_evidence,
        vs_support=vs_support,
        allowed_value_stream_names=allowed_value_stream_names,
    )

    logger.info("[SummaryRAG] Step 7 (LLM selection) done in %.2fs", time.time() - t7)
    if trace_mode:
        trace["step7_llm_selection"] = {
            "label": "LLM Value Stream Selection",
            "timing_s": round(time.time() - t7, 2),
            "system_prompt": selection_result.get("prompt_system", ""),
            "user_prompt": selection_result.get("prompt_user", ""),
            "candidates_sent_to_llm": len(selection_result.get("candidates_after_filter") or enriched_candidates),
            "raw_llm_response": selection_result.get("raw_response", ""),
            "selected_count": len(selection_result.get("selected_value_streams", [])),
            "rejected_count": len(selection_result.get("rejected_candidates", [])),
        }

    # - Step 8: Finalize selected value streams (no default canonical injection) -
    seen_names: set[str] = set()
    llm_selected: List[Dict[str, Any]] = []
    for vs in selection_result.get("selected_value_streams", []):
        cleaned_name = _clean_value_stream_name(vs.get("entity_name") or "")
        if not cleaned_name:
            continue
        vs_key = _value_stream_key(cleaned_name)
        if not vs_key or vs_key in seen_names:
            continue
        seen_names.add(vs_key)
        llm_selected.append({**vs, "entity_name": cleaned_name})

    final_selected = llm_selected

    if trace_mode:
        trace["step7_canonical_injection"] = {
            "label": "Selected Value Stream Finalization",
            "canonical_count": 0,
            "llm_selected_count": len(llm_selected),
            "llm_selected_deduplicated": [{"entity_name": v.get("entity_name"), "confidence": v.get("confidence")} for v in llm_selected],
            "canonical_injected": [],
            "final_total": len(final_selected),
        }

    elapsed = time.time() - t_start
    logger.info(
        "[SummaryRAG] DONE in %.2fs | selected=%d | analogs=%d | warnings=%s",
        elapsed, len(final_selected),
        len(analog_tickets), warnings or "none",
    )

    # Evaluation log: structured record for tuning and regression analysis
    eval_log: Dict[str, Any] = {
        "top_analogs": [
            {"ticket_id": t.get("ticket_id"), "score": t.get("score"), "rank": t.get("rank")}
            for t in analog_tickets
        ],
        "analog_scores": [t.get("score", 0.0) for t in analog_tickets],
        "kg_candidates": [c.get("entity_name") for c in candidates],
        "capability_hits": capability_mapping.get("capability_hits", []),
        "promoted_value_streams": capability_mapping.get("promoted_value_streams", []),
        "enriched_candidates": [c.get("entity_name") for c in enriched_candidates],
        "selected_value_streams": [vs.get("entity_name") for vs in final_selected],
        "raw_evidence_count": len(raw_evidence),
        "fallbacks_hit": [w for w in warnings if w.endswith("_failed") or "fallback" in w],
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
            candidates=candidates,
            capability_mapping=capability_mapping,
            raw_evidence=raw_evidence,
            selector_raw_response=selection_result.get("raw_response"),
            eval_log=eval_log,
        )

    return {
        "selected_value_streams": final_selected,
        "rejected_candidates": selection_result.get("rejected_candidates", []),
        "new_card_summary": new_card_summary,
        "analog_tickets": analog_tickets,
        "historical_value_stream_support": vs_support,
        "candidate_value_streams": [
            {
                "entity_id": c.get("entity_id", ""),
                "entity_name": c.get("entity_name", ""),
                "score": round(float(c.get("score") or c.get("best_score") or 0.0), 4),
                "raw_score": round(float(c.get("raw_score") or c.get("best_score") or c.get("score") or 0.0), 4),
                "description": (c.get("description") or "")[:200],
            }
            for c in enriched_candidates
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


def _persist_debug_artifacts(
    output_dir: str,
    *,
    new_card_summary: Dict[str, Any],
    analog_tickets: List[Dict[str, Any]],
    vs_support: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    capability_mapping: Dict[str, Any],
    raw_evidence: List[Dict[str, Any]],
    selector_raw_response: Optional[str],
    eval_log: Dict[str, Any],
) -> None:
    """
    Write debug artifacts for a single pipeline run to `output_dir`.

    Files written:
    - `new_card_summary.json`   - structured new-card summary
    - `analog_tickets.json`     - FAISS top analog results with scores
    - `vs_support.json`         - VS support aggregation output
    - `kg_candidates.json`      - KG candidate list
    - `capability_mapping.json` - capability hits + promotions + enriched list
    - `raw_evidence.json`       - raw snippet evidence
    - `selector_response.json`  - raw LLM selector output
    - `eval_log.json`           - structured evaluation log
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        artifacts = {
            "new_card_summary.json": new_card_summary,
            "analog_tickets.json": analog_tickets,
            "vs_support.json": vs_support,
            "kg_candidates.json": candidates,
            "capability_mapping.json": capability_mapping,
            "raw_evidence.json": raw_evidence,
            "selector_response.json": {"raw_response": selector_raw_response},
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
    Build a best-effort summary using keyword heuristics when LLM summarization fails.

    Extracts actors, direct functions, and domain tags from the cleaned text so
    that FAISS and KG retrieval still have structured signals rather than only raw text.
    """
    lower = cleaned_text.lower()
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]

    # Use first non-empty line as short summary proxy (often a title)
    first_line = lines[0] if lines else ""
    short_summary = first_line if len(first_line) > 20 else cleaned_text[:200]

    actors = [kw.title() for kw in ACTOR_KEYWORDS if kw in lower]
    direct_functions = [kw.title() for kw in FUNCTION_KEYWORDS if kw in lower]

    domain_tags = []
    for domain, signals in DOMAIN_KEYWORDS.items():
        if any(s in lower for s in signals):
            domain_tags.append(domain.title())

    return {
        "short_summary": short_summary,
        "business_goal": lines[1] if len(lines) > 1 else "",
        "actors": actors,
        "direct_functions": direct_functions,
        "implied_functions": [],
        "change_types": [],
        "domain_tags": domain_tags,
        "evidence_sentences": lines[:3],
    }

def _empty_result(error: str) -> Dict[str, Any]:
    return {
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
