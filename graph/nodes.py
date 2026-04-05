"""
LangGraph node functions for the V5 prediction pipeline.

Each node takes a PredictionState dict, performs its step, and returns
a partial state dict with updated keys. LangGraph merges the returned
dict into the running state.

Node order:
  clean_and_summarize
  → retrieve_analogs
  → collect_vs_evidence
  → retrieve_kg
  → map_capabilities
  → extract_card_candidates
  → collect_raw_evidence
  → build_evidence
  → fuse_scores
  → verify_candidates  (Pass 1)
  → finalize_selection (Pass 2)
  → finalize_output
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from summary_rag.models.graph_state import PredictionState
from summary_rag.ingestion.adapters import (
    LLMService,
    EmbeddingService,
    KGRetrievalService,
    get_default_llm,
    get_default_embedding,
    get_default_kg,
    clean_card_text,
    normalize_text as _norm_text,
)
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
from summary_rag.generation.candidate_evidence import (
    build_candidate_evidence,
    SOURCE_SUMMARY,
    ALL_SOURCES,
    _classify_support_type,
)
from summary_rag.generation.fusion import compute_fused_scores, apply_candidate_floor
from summary_rag.generation.card_candidates import (
    extract_summary_candidates,
    extract_chunk_candidates,
    extract_historical_footprint_candidates,
)
from summary_rag.chains.summary_chain import SummaryChain
from summary_rag.chains.selector_verify_chain import SelectorVerifyChain
from summary_rag.chains.selector_finalize_chain import SelectorFinalizeChain

logger = logging.getLogger(__name__)

_VSR_SUFFIX_RE = re.compile(r"\s*(\s*VSR[0-9A-Z-]*\s*)\s*$", re.IGNORECASE)
_EMPTY_PARENS_SUFFIX_RE = re.compile(r"\s*\(\s*\)\s*$")


def _clean_vs_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return ""
    cleaned = _VSR_SUFFIX_RE.sub("", cleaned)
    cleaned = _EMPTY_PARENS_SUFFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"\(-\)", "", cleaned).strip(" -")
    return cleaned


def _normalize_kg_score(candidate: Dict[str, Any]) -> float:
    raw = float(candidate.get("score") or candidate.get("best_score") or 0.0)
    if raw <= 1.0:
        return max(0.0, raw)
    return min(1.0, raw / (raw + 10.0))


def _inject_summary_candidates(
    candidate_evidence: List[Dict[str, Any]],
    summary_candidates: List[Dict[str, Any]],
) -> None:
    """Merge summary-source candidates into existing CandidateEvidence objects."""
    by_name = {_norm_text(c.get("candidate_name", "")): c for c in candidate_evidence}

    for sc in summary_candidates:
        name = (sc.get("entity_name") or "").strip()
        key = _norm_text(name)
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

    for entry in candidate_evidence:
        entry["support_type"] = _classify_support_type(entry)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_clean_and_summarize(state: PredictionState) -> Dict[str, Any]:
    """Step 1: Clean card text and generate semantic summary."""
    t = time.time()
    raw_text = state.get("raw_text", "")
    cleaned_text = clean_card_text(raw_text)

    if not cleaned_text.strip():
        logger.warning("[node_clean_and_summarize] Empty text after cleaning")
        errors = list(state.get("errors", []))
        errors.append("empty_input_text")
        return {"cleaned_text": cleaned_text, "errors": errors}

    llm: Optional[LLMService] = state.get("_llm")  # type: ignore[assignment]
    chain = SummaryChain(llm=llm)

    try:
        new_card_summary = chain.run_card(card_text=cleaned_text)
    except Exception as exc:
        logger.error("[node_clean_and_summarize] Summary failed: %s", exc)
        new_card_summary = _deterministic_fallback_summary(cleaned_text)
        warnings = list(state.get("warnings", []))
        warnings.append(f"summary_generation_failed: {type(exc).__name__}")
        return {
            "cleaned_text": cleaned_text,
            "new_card_summary": new_card_summary,
            "warnings": warnings,
        }

    logger.info("[node_clean_and_summarize] done in %.2fs", time.time() - t)
    return {"cleaned_text": cleaned_text, "new_card_summary": new_card_summary}


def node_retrieve_analogs(state: PredictionState) -> Dict[str, Any]:
    """Step 2: Search FAISS for top analog historical tickets."""
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    index_dir = state.get("_index_dir", DEFAULT_INDEX_DIR)  # type: ignore[call-overload]
    top_k = state.get("top_k_analogs", 5)

    analog_tickets: List[Dict[str, Any]] = []
    try:
        analog_tickets = retrieve_analog_tickets(
            new_card_summary,
            index_dir=index_dir,
            top_k=top_k,
        )
    except Exception as exc:
        logger.error("[node_retrieve_analogs] FAISS search failed: %s", exc)
        warnings = list(state.get("warnings", []))
        warnings.append(f"faiss_search_failed: {type(exc).__name__}")
        return {"analog_tickets": [], "warnings": warnings}

    logger.info("[node_retrieve_analogs] done in %.2fs | %d analogs", time.time() - t, len(analog_tickets))
    return {"analog_tickets": analog_tickets}


def node_collect_vs_evidence(state: PredictionState) -> Dict[str, Any]:
    """Step 3: Collect VS evidence from analog tickets."""
    t = time.time()
    analog_tickets = state.get("analog_tickets", [])
    ticket_chunks_dir = state.get("_ticket_chunks_dir", "ticket_chunks")  # type: ignore[call-overload]

    vs_support = collect_value_stream_evidence(
        analog_tickets,
        ticket_chunks_dir=ticket_chunks_dir,
    )
    logger.info("[node_collect_vs_evidence] done in %.2fs | %d VS entries", time.time() - t, len(vs_support))
    return {"historical_value_stream_support": vs_support}


def node_retrieve_kg(state: PredictionState) -> Dict[str, Any]:
    """Step 4: Retrieve KG candidate value streams."""
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    cleaned_text = state.get("cleaned_text", "")
    allowed_names = state.get("allowed_value_stream_names")
    top_k = state.get("_top_kg_candidates", 20)  # type: ignore[call-overload]

    from summary_rag.ingestion import build_retrieval_text
    kg_query_text = build_retrieval_text(new_card_summary).strip() or cleaned_text

    kg_candidates: List[Dict[str, Any]] = []
    try:
        kg_candidates = retrieve_kg_candidates(
            kg_query_text,
            top_k=top_k,
            allowed_names=allowed_names,
        )
    except Exception as exc:
        logger.error("[node_retrieve_kg] KG retrieval failed: %s", exc)
        warnings = list(state.get("warnings", []))
        warnings.append(f"kg_retrieval_failed: {type(exc).__name__}")
        return {"kg_candidates": [], "warnings": warnings}

    logger.info("[node_retrieve_kg] done in %.2fs | %d candidates", time.time() - t, len(kg_candidates))
    return {"kg_candidates": kg_candidates}


def node_map_capabilities(state: PredictionState) -> Dict[str, Any]:
    """Step 5: Capability mapping and candidate enrichment."""
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    cleaned_text = state.get("cleaned_text", "")
    vs_support = state.get("historical_value_stream_support", [])
    kg_candidates = state.get("kg_candidates", [])
    allowed_names = state.get("allowed_value_stream_names")

    capability_mapping = map_capabilities_to_candidates(
        new_card_summary=new_card_summary,
        cleaned_text=cleaned_text,
        vs_support=vs_support,
        candidates=kg_candidates,
        allowed_value_stream_names=allowed_names,
    )
    enriched_candidates = capability_mapping.get("enriched_candidates", kg_candidates)
    logger.info(
        "[node_map_capabilities] done in %.2fs | hits=%d | promoted=%d",
        time.time() - t,
        len(capability_mapping.get("capability_hits", [])),
        len(capability_mapping.get("promoted_value_streams", [])),
    )
    return {
        "capability_mapping": capability_mapping,
        "enriched_candidates": enriched_candidates,
    }


def node_extract_card_candidates(state: PredictionState) -> Dict[str, Any]:
    """Step 5b: Extract summary/chunk/footprint candidates from the card itself."""
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    cleaned_text = state.get("cleaned_text", "")
    analog_tickets = state.get("analog_tickets", [])
    allowed_names = state.get("allowed_value_stream_names")

    allowed_set = set(allowed_names) if allowed_names else None
    summary_candidates = extract_summary_candidates(new_card_summary, allowed_names=allowed_set)
    chunk_candidates = extract_chunk_candidates(cleaned_text, allowed_names=allowed_set)
    footprint_candidates = extract_historical_footprint_candidates(analog_tickets, allowed_names=allowed_set)

    logger.info(
        "[node_extract_card_candidates] done in %.2fs | summary=%d | chunk=%d | footprint=%d",
        time.time() - t, len(summary_candidates), len(chunk_candidates), len(footprint_candidates),
    )
    return {
        "summary_candidates": summary_candidates,
        "chunk_candidates": chunk_candidates,
        "historical_footprint_candidates": footprint_candidates,
    }


def node_collect_raw_evidence(state: PredictionState) -> Dict[str, Any]:
    """Step 6: Collect raw evidence chunks (must run before build_evidence)."""
    include_raw = state.get("_include_raw_evidence", True)  # type: ignore[call-overload]
    analog_tickets = state.get("analog_tickets", [])
    if not include_raw or not analog_tickets:
        return {"raw_evidence": [], "attachment_candidates": []}

    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    cleaned_text = state.get("cleaned_text", "")
    ticket_chunks_dir = state.get("_ticket_chunks_dir", "ticket_chunks")  # type: ignore[call-overload]
    max_tickets = state.get("_max_raw_evidence_tickets", 3)  # type: ignore[call-overload]

    from summary_rag.ingestion import build_retrieval_text
    kg_query = build_retrieval_text(new_card_summary).strip() or cleaned_text

    top_ticket_ids = [
        t_["ticket_id"] for t_ in analog_tickets[:max_tickets]
        if t_.get("ticket_id")
    ]
    raw_evidence: List[Dict[str, Any]] = []
    try:
        raw_evidence = retrieve_raw_evidence_for_tickets(
            top_ticket_ids,
            ticket_chunks_dir=ticket_chunks_dir,
            query_text=kg_query,
        )
    except Exception as exc:
        logger.warning("[node_collect_raw_evidence] Failed: %s", exc)

    attachment_candidates = collect_attachment_candidates(raw_evidence, analog_tickets)
    logger.info(
        "[node_collect_raw_evidence] done in %.2fs | %d snippets | %d attachment candidates",
        time.time() - t, len(raw_evidence), len(attachment_candidates),
    )
    return {"raw_evidence": raw_evidence, "attachment_candidates": attachment_candidates}


def node_build_evidence(state: PredictionState) -> Dict[str, Any]:
    """Step 7: Build CandidateEvidence objects from all sources."""
    t = time.time()
    vs_support = state.get("historical_value_stream_support", [])
    kg_candidates = state.get("kg_candidates", [])
    capability_mapping = state.get("capability_mapping", {})
    chunk_candidates = state.get("chunk_candidates", [])
    attachment_candidates = state.get("attachment_candidates", [])
    footprint_candidates = state.get("historical_footprint_candidates", [])
    summary_candidates = state.get("summary_candidates", [])

    historical_candidates = [
        {
            "entity_name": s.get("entity_name", ""),
            "score": float(s.get("best_score") or 0.0),
            "supporting_evidence": s.get("supporting_evidence", []) or s.get("supporting_functions", []),
            "capability_tags": s.get("capability_tags", []),
            "operational_footprint": s.get("operational_footprint", []),
        }
        for s in vs_support
    ]

    kg_for_evidence = [
        {
            "entity_name": c.get("entity_name", ""),
            "entity_id": c.get("entity_id", ""),
            "score": _normalize_kg_score(c),
            "description": (c.get("description") or "")[:200],
        }
        for c in kg_candidates
    ]

    all_historical = historical_candidates + list(footprint_candidates)

    candidate_evidence = build_candidate_evidence(
        kg_candidates=kg_for_evidence,
        historical_candidates=all_historical,
        capability_candidates=capability_mapping.get("capability_candidates", []),
        chunk_candidates=chunk_candidates,
        attachment_candidates=attachment_candidates,
    )
    _inject_summary_candidates(candidate_evidence, summary_candidates)

    logger.info(
        "[node_build_evidence] done in %.2fs | %d candidates",
        time.time() - t, len(candidate_evidence),
    )
    return {"candidate_evidence": candidate_evidence}


def node_fuse_scores(state: PredictionState) -> Dict[str, Any]:
    """Step 8: Source-aware fused ranking + candidate floor."""
    t = time.time()
    candidate_evidence = list(state.get("candidate_evidence", []))
    min_floor = state.get("_min_candidate_floor", 8)  # type: ignore[call-overload]

    fused = compute_fused_scores(candidate_evidence)
    fused = apply_candidate_floor(fused, min_candidates=min_floor)

    logger.info("[node_fuse_scores] done in %.2fs | %d candidates", time.time() - t, len(fused))
    return {"fused_candidates": fused}


def node_verify_candidates(state: PredictionState) -> Dict[str, Any]:
    """Step 9 (Pass 1): LLM evidence verification."""
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    analog_tickets = state.get("analog_tickets", [])
    fused_candidates = state.get("fused_candidates", [])
    raw_evidence = state.get("raw_evidence", [])
    allowed_names = state.get("allowed_value_stream_names")
    llm: Optional[LLMService] = state.get("_llm")  # type: ignore[assignment]

    # Filter to allowed names
    candidates = fused_candidates
    if allowed_names:
        allowed_set = {_norm_text(n) for n in allowed_names}
        candidates = [
            c for c in fused_candidates
            if _norm_text(c.get("candidate_name") or c.get("entity_name", "")) in allowed_set
        ]

    chain = SelectorVerifyChain(llm=llm)
    result = chain.run(
        new_card_summary=new_card_summary,
        analog_tickets=analog_tickets,
        candidates=candidates,
        raw_evidence=raw_evidence,
    )

    logger.info("[node_verify_candidates] done in %.2fs", time.time() - t)
    return {
        "selection_result": result,
        "_verify_result": result,  # pass to finalize node
    }


def node_finalize_selection(state: PredictionState) -> Dict[str, Any]:
    """Step 9 (Pass 2): LLM finalize and calibrate."""
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    preliminary = state.get("_verify_result") or state.get("selection_result", {})
    llm: Optional[LLMService] = state.get("_llm")  # type: ignore[assignment]

    chain = SelectorFinalizeChain(llm=llm)
    result = chain.run(
        new_card_summary=new_card_summary,
        preliminary_classification=preliminary,
    )

    logger.info("[node_finalize_selection] done in %.2fs", time.time() - t)
    return {"selection_result": result}


def node_finalize_output(state: PredictionState) -> Dict[str, Any]:
    """Step 10: Finalize three-class output with dedup and compat fields."""
    selection_result = state.get("selection_result", {})
    seen_names: set = set()

    def _dedup_vs_list(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for vs in items:
            cleaned = _clean_vs_name(vs.get("entity_name") or "")
            if not cleaned:
                continue
            key = _norm_text(cleaned)
            if not key or key in seen_names:
                continue
            seen_names.add(key)
            out.append({**vs, "entity_name": cleaned})
        return out

    directly_supported = _dedup_vs_list(selection_result.get("directly_supported", []))
    pattern_inferred = _dedup_vs_list(selection_result.get("pattern_inferred", []))
    no_evidence = selection_result.get("no_evidence", [])

    selected_value_streams = []
    for vs in directly_supported:
        selected_value_streams.append({
            "entity_id": vs.get("entity_id", ""),
            "entity_name": vs.get("entity_name", ""),
            "confidence": vs.get("confidence", 0.8),
            "reason": vs.get("evidence", ""),
            "support_type": "direct",
        })
    for vs in pattern_inferred:
        selected_value_streams.append({
            "entity_id": vs.get("entity_id", ""),
            "entity_name": vs.get("entity_name", ""),
            "confidence": vs.get("confidence", 0.6),
            "reason": vs.get("evidence", ""),
            "support_type": "pattern",
        })

    rejected_candidates = [
        {"entity_name": vs.get("entity_name", ""), "reason": vs.get("reason", "No evidence")}
        for vs in no_evidence
    ]

    return {
        "directly_supported": directly_supported,
        "pattern_inferred": pattern_inferred,
        "no_evidence": no_evidence,
        "selected_value_streams": selected_value_streams,
        "rejected_candidates": rejected_candidates,
    }


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

_ACTOR_KEYWORDS = [
    "member", "provider", "broker", "employer", "plan sponsor",
    "internal ops", "clinical", "care manager", "agent", "beneficiary",
]

_FUNCTION_KEYWORDS = [
    "claims", "enrollment", "billing", "prior auth", "authorization",
    "eligibility", "referral", "appeals", "grievance", "pharmacy",
    "care management", "utilization", "network", "credentialing",
    "payment", "risk adjustment", "quality", "reporting",
]

_DOMAIN_KEYWORDS = {
    "clinical": ["clinical", "care", "health", "medical", "diagnosis", "treatment"],
    "financial": ["billing", "claims", "payment", "cost", "revenue", "finance"],
    "operational": ["process", "workflow", "efficiency", "ops", "operational"],
    "it": ["system", "platform", "integration", "api", "migration", "data", "digital"],
    "regulatory": ["regulatory", "compliance", "hipaa", "cms", "mandate", "audit"],
}


def _deterministic_fallback_summary(cleaned_text: str) -> Dict[str, Any]:
    lower = cleaned_text.lower()
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
    first_line = lines[0] if lines else ""
    short_summary = first_line if len(first_line) > 20 else cleaned_text[:200]
    actors = [kw.title() for kw in _ACTOR_KEYWORDS if kw in lower]
    raw_functions = [kw.title() for kw in _FUNCTION_KEYWORDS if kw in lower]
    canonical_functions = normalize_functions(raw_functions)
    domain_tags = [
        domain.title() for domain, signals in _DOMAIN_KEYWORDS.items()
        if any(s in lower for s in signals)
    ]
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
