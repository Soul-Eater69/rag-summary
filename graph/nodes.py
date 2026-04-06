"""
LangGraph node functions for the V5 prediction pipeline.

Each node takes a PredictionState dict, performs its step, and returns
a partial state dict with updated keys. LangGraph merges the returned
dict into the running state.

Node order (V6 — 14 nodes):
  clean_and_summarize
  → retrieve_analogs
  → collect_vs_evidence      (+ bundle_patterns, downstream_chains)
  → retrieve_kg
  → retrieve_themes          (auto-discovers FAISS index if present)
  → map_capabilities
  → extract_card_candidates
  → collect_raw_evidence
  → parse_attachments        (V6: section-level parsing → attachment_native_candidates)
  → build_evidence           (merges all 7 sources)
  → fuse_scores
  → verify_candidates        (Pass 1)
  → finalize_selection       (Pass 2)
  → finalize_output
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from rag_summary.models.graph_state import PredictionState
from rag_summary.ingestion.adapters import (
    LLMService,
    EmbeddingService,
    KGRetrievalService,
    get_default_llm,
    get_default_embedding,
    get_default_kg,
    clean_card_text,
    normalize_text as _norm_text,
)
from rag_summary.ingestion.function_normalizer import normalize_functions
from rag_summary.ingestion.faiss_indexer import DEFAULT_INDEX_DIR
from rag_summary.retrieval import (
    retrieve_analog_tickets,
    retrieve_raw_evidence_for_tickets,
    retrieve_kg_candidates,
    collect_value_stream_evidence,
    collect_attachment_candidates,
    detect_bundle_patterns,
    detect_downstream_chains,
)
from rag_summary.generation.capability_mapper import map_capabilities_to_candidates
from rag_summary.generation.candidate_evidence import (
    build_candidate_evidence,
    SOURCE_SUMMARY,
    ALL_SOURCES,
    _classify_support_type,
)
from rag_summary.generation.fusion import compute_fused_scores, apply_candidate_floor
from rag_summary.generation.card_candidates import (
    extract_summary_candidates,
    extract_chunk_candidates,
    extract_historical_footprint_candidates,
    extract_card_attachment_candidates,
)
from rag_summary.retrieval import enrich_historical_candidates
from rag_summary.ingestion.adapters import get_default_theme, ThemeRetrievalService
from rag_summary.ingestion.attachment_parser import AttachmentParser
from rag_summary.generation.attachment_candidates import extract_attachment_native_candidates
from rag_summary.chains.summary_chain import SummaryChain
from rag_summary.chains.selector_verify_chain import SelectorVerifyChain
from rag_summary.chains.selector_finalize_chain import SelectorFinalizeChain
from rag_summary.models.summary_doc import SummaryDoc
from rag_summary.models.candidate_judgment import CandidateJudgment, VerificationResult
from rag_summary.models.selection import SelectionResult, SupportedStream, UnsupportedStream
from rag_summary.generation.candidate_evidence import SOURCE_THEME

logger = logging.getLogger(__name__)

# Theme source is wired as a first-class graph node (node_retrieve_themes).
# By default it uses _NoopThemeService (returns []) so the theme slot stays at 0.
# To activate: pass a ThemeRetrievalService implementation via the _theme_svc
# key in initial state or through run_prediction_graph(theme_svc=...).
_THEME_DEFAULT_INACTIVE = True  # True = no real theme service configured yet

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


def _inject_footprint_patterns(
    candidate_evidence: List[Dict[str, Any]],
    bundle_patterns: List[Dict[str, Any]],
    downstream_chains: List[Dict[str, Any]],
) -> None:
    """
    V6: Inject bundle pattern and downstream chain evidence snippets into existing
    CandidateEvidence objects. Increments diagnostic counters.

    Bundle patterns: for each (primary_vs, bundled_vs) pair, if the bundled_vs
    is an existing candidate, add a sub_source=bundle_pattern snippet to it.

    Downstream chains: for each (upstream_vs, downstream_vs) pair, if the
    downstream_vs is an existing candidate, add a sub_source=downstream_chain snippet.
    """
    from rag_summary.generation.candidate_evidence import SOURCE_HISTORICAL
    by_name = {_norm_text(c.get("candidate_name", "")): c for c in candidate_evidence}

    for bp in bundle_patterns:
        bundled = (bp.get("bundled_vs") or "").strip()
        primary = (bp.get("primary_vs") or "").strip()
        key = _norm_text(bundled)
        entry = by_name.get(key)
        if not entry:
            continue
        fraction = bp.get("co_occurrence_fraction", 0.0)
        count = bp.get("co_occurrence_count", 0)
        snippet_score = round(0.20 * fraction, 4)
        entry.setdefault("evidence_snippets", []).append({
            "source": SOURCE_HISTORICAL,
            "snippet": (
                f"Bundle co-occurrence: appears with '{primary}' in "
                f"{fraction:.0%} of similar analogs ({count} tickets)"
            ),
            "score": snippet_score,
            "sub_source": "bundle_pattern",
        })
        entry["bundle_pattern_count"] = entry.get("bundle_pattern_count", 0) + 1
        # V6: high co-occurrence fraction also boosts the historical source score
        # so fusion arithmetic reflects the pattern, not just the LLM prompt context
        if fraction >= 0.60:
            old_score = entry.get("source_scores", {}).get(SOURCE_HISTORICAL, 0.0)
            boost = round(0.12 * fraction, 4)  # up to +0.084 at fraction=0.70
            entry.setdefault("source_scores", {})[SOURCE_HISTORICAL] = min(
                1.0, old_score + boost
            )

    for dc in downstream_chains:
        down_vs = (dc.get("downstream_vs") or "").strip()
        up_vs = (dc.get("upstream_vs") or "").strip()
        key = _norm_text(down_vs)
        entry = by_name.get(key)
        if not entry:
            continue
        analog_count = dc.get("analog_count", 0)
        entry.setdefault("evidence_snippets", []).append({
            "source": SOURCE_HISTORICAL,
            "snippet": (
                f"Downstream activation: '{up_vs}' typically leads to this "
                f"stream in {analog_count} analog(s)"
            ),
            "score": 0.18,
            "sub_source": "downstream_chain",
        })
        entry["downstream_chain_count"] = entry.get("downstream_chain_count", 0) + 1
        # V6: downstream chain presence always adds a small historical score boost
        old_score = entry.get("source_scores", {}).get(SOURCE_HISTORICAL, 0.0)
        entry.setdefault("source_scores", {})[SOURCE_HISTORICAL] = min(
            1.0, old_score + 0.08
        )


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_clean_and_summarize(state: PredictionState) -> Dict[str, Any]:
    """Step 1: Clean card text and generate semantic summary. Returns SummaryDoc."""
    t = time.time()
    raw_text = state.get("raw_text", "")
    cleaned_text = clean_card_text(raw_text)

    if not cleaned_text.strip():
        logger.warning("[node_clean_and_summarize] Empty text after cleaning")
        errors = list(state.get("errors", []))
        errors.append("empty_input_text")
        return {"cleaned_text": cleaned_text, "errors": errors}

    if _THEME_DEFAULT_INACTIVE and not state.get("_theme_svc"):  # type: ignore[call-overload]
        logger.debug(
            "[node_clean_and_summarize] Theme source uses noop default — "
            "pass _theme_svc to activate (see ThemeRetrievalService in adapters.py)"
        )

    llm: Optional[LLMService] = state.get("_llm")  # type: ignore[assignment]
    chain = SummaryChain(llm=llm)

    try:
        summary_doc: SummaryDoc = chain.run_card(card_text=cleaned_text)
    except Exception as exc:
        logger.error("[node_clean_and_summarize] Summary failed: %s", exc)
        fb = _deterministic_fallback_summary(cleaned_text)
        try:
            summary_doc = SummaryDoc(**fb)
        except Exception:
            summary_doc = SummaryDoc()
        warnings = list(state.get("warnings", []))
        warnings.append(f"summary_generation_failed: {type(exc).__name__}")
        return {
            "cleaned_text": cleaned_text,
            "new_card_summary": summary_doc.model_dump(),
            "warnings": warnings,
        }

    logger.info("[node_clean_and_summarize] done in %.2fs", time.time() - t)
    # Serialize to dict for state transport; nodes consume from state as dicts
    return {"cleaned_text": cleaned_text, "new_card_summary": summary_doc.model_dump()}


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
    """Step 3: Collect VS evidence from analog tickets.

    V6: Also detects bundle patterns and downstream chains across the analog set.
    Bundle patterns: VS pairs that co-occur in ≥60% of analogs.
    Downstream chains: VS entries consistently activated downstream of direct ones.
    """
    t = time.time()
    analog_tickets = state.get("analog_tickets", [])
    ticket_chunks_dir = state.get("_ticket_chunks_dir", "ticket_chunks")  # type: ignore[call-overload]
    allowed_names = state.get("allowed_value_stream_names")

    vs_support = collect_value_stream_evidence(
        analog_tickets,
        ticket_chunks_dir=ticket_chunks_dir,
    )

    # V6: detect historical footprint patterns
    bundle_patterns = detect_bundle_patterns(
        analog_tickets,
        min_co_occurrence_fraction=0.60,
        min_analog_count=2,
        allowed_names=list(allowed_names) if allowed_names else None,
    )
    downstream_chains = detect_downstream_chains(vs_support)

    logger.info(
        "[node_collect_vs_evidence] done in %.2fs | %d VS entries | "
        "%d bundle patterns | %d downstream chains",
        time.time() - t, len(vs_support), len(bundle_patterns), len(downstream_chains),
    )
    return {
        "historical_value_stream_support": vs_support,
        "bundle_patterns": bundle_patterns,
        "downstream_chains": downstream_chains,
    }


def node_retrieve_kg(state: PredictionState) -> Dict[str, Any]:
    """Step 4: Retrieve KG candidate value streams."""
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    cleaned_text = state.get("cleaned_text", "")
    allowed_names = state.get("allowed_value_stream_names")
    top_k = state.get("_top_kg_candidates", 20)  # type: ignore[call-overload]

    from rag_summary.ingestion import build_retrieval_text
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


def node_retrieve_themes(state: PredictionState) -> Dict[str, Any]:
    """
    Step 4b: Retrieve theme-cluster candidates via ThemeRetrievalService.

    Inactive by default (_NoopThemeService returns []). To activate:
      - Implement ThemeRetrievalService (see ingestion/adapters.py)
      - Pass it as _theme_svc in the initial state or via run_prediction_graph()

    When active, theme candidates feed directly into the SOURCE_THEME slot in
    CandidateEvidence, adding a seventh independent evidence dimension.
    """
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    cleaned_text = state.get("cleaned_text", "")
    allowed_names = state.get("allowed_value_stream_names")
    # V6: intake_date cutoff prevents leaking future themes of this card
    intake_date = state.get("_intake_date")  # type: ignore[call-overload]

    theme_svc = state.get("_theme_svc")  # type: ignore[call-overload]
    if theme_svc is None:
        # Auto-discover: load FaissThemeRetrievalService if theme index files exist
        import pathlib as _pathlib
        _default_theme_dir = str(
            _pathlib.Path(__file__).resolve().parent.parent / "config" / "theme_index"
        )
        _theme_index_file = _pathlib.Path(_default_theme_dir) / "theme_index.faiss"
        if _theme_index_file.exists():
            try:
                from rag_summary.ingestion.theme_retrieval_service import FaissThemeRetrievalService
                from rag_summary.ingestion.adapters import get_default_embedding
                theme_svc = FaissThemeRetrievalService(
                    theme_index_dir=_default_theme_dir,
                    embedding_svc=get_default_embedding(),
                )
                logger.info(
                    "[node_retrieve_themes] Auto-loaded FaissThemeRetrievalService from %s",
                    _default_theme_dir,
                )
            except Exception as _exc:
                logger.warning("[node_retrieve_themes] Auto-load failed: %s", _exc)
                theme_svc = None
    if theme_svc is None:
        # Final fallback: keyword-based theme service — always active, no setup needed
        from rag_summary.ingestion.keyword_theme_service import KeywordThemeService
        theme_svc = KeywordThemeService()
        logger.debug(
            "[node_retrieve_themes] No theme index found — using KeywordThemeService fallback"
        )

    from rag_summary.ingestion import build_retrieval_text
    query = build_retrieval_text(new_card_summary).strip() or cleaned_text

    theme_candidates: List[Dict[str, Any]] = []
    try:
        import inspect as _inspect
        _sig = _inspect.signature(theme_svc.retrieve_theme_candidates)  # type: ignore[union-attr]
        _kwargs: Dict[str, Any] = {
            "top_k": 10,
            "allowed_names": list(allowed_names) if allowed_names else None,
        }
        if "cutoff_date" in _sig.parameters and intake_date:
            _kwargs["cutoff_date"] = intake_date
        raw = theme_svc.retrieve_theme_candidates(query, **_kwargs)  # type: ignore[union-attr]
        theme_candidates = raw or []
    except Exception as exc:
        logger.warning("[node_retrieve_themes] Failed: %s", exc)

    if theme_candidates:
        logger.info(
            "[node_retrieve_themes] done in %.2fs | %d theme candidates",
            time.time() - t, len(theme_candidates),
        )
    else:
        logger.debug(
            "[node_retrieve_themes] inactive — 0 candidates "
            "(wire a ThemeRetrievalService via _theme_svc to activate)"
        )

    return {"theme_candidates": theme_candidates}


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
    """
    Step 5b: Extract card-native candidates from the new card's own content.

    Four extraction paths (all LLM-free):
    - summary_candidates: capability_tags + canonical functions → capability map
    - chunk_candidates: cue term scan over cleaned card text → capability map
    - card_attachment_candidates: attachment indicators × domain cues → attachment source
    - footprint_candidates: analog capability_tags × capability map → pattern signals
    """
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    cleaned_text = state.get("cleaned_text", "")
    analog_tickets = state.get("analog_tickets", [])
    allowed_names = state.get("allowed_value_stream_names")

    allowed_set = set(allowed_names) if allowed_names else None
    summary_candidates = extract_summary_candidates(new_card_summary, allowed_names=allowed_set)
    chunk_candidates = extract_chunk_candidates(cleaned_text, allowed_names=allowed_set)
    card_attachment_candidates = extract_card_attachment_candidates(cleaned_text, allowed_names=allowed_set)
    footprint_candidates = extract_historical_footprint_candidates(analog_tickets, allowed_names=allowed_set)

    logger.info(
        "[node_extract_card_candidates] done in %.2fs | "
        "summary=%d | chunk=%d | card_attach=%d | footprint=%d",
        time.time() - t,
        len(summary_candidates), len(chunk_candidates),
        len(card_attachment_candidates), len(footprint_candidates),
    )
    return {
        "summary_candidates": summary_candidates,
        "chunk_candidates": chunk_candidates,
        "card_attachment_candidates": card_attachment_candidates,
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

    from rag_summary.ingestion import build_retrieval_text
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


def node_parse_attachments(state: PredictionState) -> Dict[str, Any]:
    """
    Step 6b (V6): Parse attachment content into structured sections.

    Two input modes:
    1. Explicit attachment files: reads _attachment_contents from state —
       a list of {"filename": str, "text": str} dicts pre-extracted by the caller.
    2. Card text fallback: scans cleaned_text for exhibit/appendix/table/scope
       structural markers; produces sections from the card body itself.

    Outputs:
      attachment_docs: List[ParsedAttachment.to_dict()] — section-level parsed docs
      attachment_native_candidates: List[candidate dicts] with sub_source="attachment_native"
    """
    t = time.time()
    cleaned_text = state.get("cleaned_text", "")
    allowed_names = state.get("allowed_value_stream_names")
    attachment_contents = state.get("_attachment_contents") or []  # type: ignore[call-overload]

    parser = AttachmentParser()
    parsed_docs = []

    # Mode 1: explicit attachment files provided by caller
    for attach in attachment_contents:
        filename = attach.get("filename", "attachment")
        text = attach.get("text", "")
        if not text.strip():
            continue
        doc = parser.parse_attachment_content(filename, text)
        parsed_docs.append(doc)

    # Mode 2: fallback — extract structural sections from the card text itself
    if not parsed_docs and cleaned_text.strip():
        card_doc = parser.parse_card_text(cleaned_text)
        if card_doc:
            parsed_docs.append(card_doc)

    attachment_docs = [doc.to_dict() for doc in parsed_docs]

    allowed_set = set(allowed_names) if allowed_names else None
    attachment_native_candidates = extract_attachment_native_candidates(
        attachment_docs,
        allowed_names=allowed_set,
    )

    section_count = sum(len(d.get("sections", [])) for d in attachment_docs)
    logger.info(
        "[node_parse_attachments] done in %.2fs | %d docs | %d sections | %d native candidates",
        time.time() - t, len(parsed_docs), section_count, len(attachment_native_candidates),
    )
    return {
        "attachment_docs": attachment_docs,
        "attachment_native_candidates": attachment_native_candidates,
    }


def node_build_evidence(state: PredictionState) -> Dict[str, Any]:
    """
    Step 7: Build CandidateEvidence objects from all 7 sources.

    Historical candidates are now enriched with support-type-weighted scores
    and operational evidence phrases before being merged into CandidateEvidence.

    Source mapping:
      kg          ← KG retrieval candidates
      historical  ← vs_support (type-weighted) + footprint candidates
      capability  ← capability_mapping candidates
      chunk       ← chunk_candidates (card text cue scan)
      attachment  ← card_attachment_candidates (card-native) + analog proxy candidates
      summary     ← summary_candidates (injected separately)
      theme       ← theme_candidates (live if ThemeRetrievalService is wired)
    """
    t = time.time()
    vs_support = state.get("historical_value_stream_support", [])
    kg_candidates = state.get("kg_candidates", [])
    capability_mapping = state.get("capability_mapping", {})
    chunk_candidates = state.get("chunk_candidates", [])
    # Merge all attachment candidates: native (parsed sections) > heuristic (card text) > proxy (analog)
    native_attach = state.get("attachment_native_candidates", [])
    card_attach = state.get("card_attachment_candidates", [])
    analog_attach = state.get("attachment_candidates", [])
    all_attachment = (
        native_attach
        + card_attach
        + [a for a in analog_attach if a.get("entity_name")]
    )
    footprint_candidates = state.get("historical_footprint_candidates", [])
    summary_candidates = state.get("summary_candidates", [])
    theme_candidates = state.get("theme_candidates", [])

    # V6: pass new card summary + analog dicts for capability overlap scoring
    new_card_summary = state.get("new_card_summary", {})
    analog_tickets = state.get("analog_tickets", [])

    enriched_vs_support = enrich_historical_candidates(
        vs_support,
        new_card_summary=new_card_summary,
        analog_summaries=analog_tickets,   # plain dicts; enrich_historical handles them
    )
    historical_candidates = [
        {
            "entity_name": s.get("entity_name", ""),
            "score": float(s.get("score") or 0.0),
            "sub_source": None,   # historical candidates have no attachment sub_source
            "supporting_evidence": (
                s.get("evidence_phrases", [])
                or s.get("supporting_evidence", [])
                or s.get("supporting_functions", [])
            ),
            "capability_tags": s.get("capability_tags", []),
            "operational_footprint": s.get("operational_footprint", []),
            "capability_overlap_score": s.get("capability_overlap_score", 0.0),
        }
        for s in enriched_vs_support
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
        attachment_candidates=all_attachment,
        theme_candidates=theme_candidates,
    )
    _inject_summary_candidates(candidate_evidence, summary_candidates)

    # V6: inject bundle pattern and downstream chain evidence snippets
    bundle_patterns = state.get("bundle_patterns", [])
    downstream_chains = state.get("downstream_chains", [])
    _inject_footprint_patterns(candidate_evidence, bundle_patterns, downstream_chains)

    active_theme = len(theme_candidates) > 0
    logger.info(
        "[node_build_evidence] done in %.2fs | %d candidates | "
        "theme_active=%s | native_attach=%d | card_attach=%d | analog_attach=%d | "
        "bundles=%d | downstream=%d",
        time.time() - t, len(candidate_evidence),
        active_theme, len(native_attach), len(card_attach), len(analog_attach),
        len(bundle_patterns), len(downstream_chains),
    )
    if not active_theme:
        logger.debug(
            "[node_build_evidence] theme source inactive — wire ThemeRetrievalService to enable"
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
    """Step 9 (Pass 1): Per-candidate evidence verification → VerificationResult."""
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    analog_tickets = state.get("analog_tickets", [])
    fused_candidates = state.get("fused_candidates", [])
    raw_evidence = state.get("raw_evidence", [])
    allowed_names = state.get("allowed_value_stream_names")
    llm: Optional[LLMService] = state.get("_llm")  # type: ignore[assignment]

    candidates = fused_candidates
    if allowed_names:
        allowed_set = {_norm_text(n) for n in allowed_names}
        candidates = [
            c for c in fused_candidates
            if _norm_text(c.get("candidate_name") or c.get("entity_name", "")) in allowed_set
        ]

    chain = SelectorVerifyChain(llm=llm)  # defaults to v3 prompt
    verification_result: VerificationResult = chain.run(
        new_card_summary=new_card_summary,
        analog_tickets=analog_tickets,
        candidates=candidates,
        raw_evidence=raw_evidence,
    )

    # Serialise to dicts for state transport
    judgment_dicts = [j.model_dump() for j in verification_result.judgments]
    logger.info("[node_verify_candidates] done in %.2fs | %d judgments", time.time() - t, len(judgment_dicts))
    return {"verify_judgments": judgment_dicts}


def node_finalize_selection(state: PredictionState) -> Dict[str, Any]:
    """Step 9 (Pass 2): VerificationResult → SelectionResult."""
    t = time.time()
    new_card_summary = state.get("new_card_summary", {})
    judgment_dicts = state.get("verify_judgments", [])
    fused_candidates = state.get("fused_candidates", [])
    llm: Optional[LLMService] = state.get("_llm")  # type: ignore[assignment]

    judgments = [
        CandidateJudgment(**j) if isinstance(j, dict) else j
        for j in judgment_dicts
    ]
    verification_result = VerificationResult(judgments=judgments)

    chain = SelectorFinalizeChain(llm=llm)  # defaults to v3 prompt
    selection_result: SelectionResult = chain.run(
        new_card_summary=new_card_summary,
        verification_result=verification_result,
        fused_candidates=fused_candidates,
    )

    logger.info("[node_finalize_selection] done in %.2fs", time.time() - t)
    # Serialise to dict for state transport; finalize_output deserialises
    return {"selection_result": selection_result.model_dump()}


def node_finalize_output(state: PredictionState) -> Dict[str, Any]:
    """Step 10: Dedup, compat fields, and finalize three-class output."""
    from rag_summary.chains.selector_finalize_chain import _judgments_to_selection_result

    selection_dict = state.get("selection_result") or {}
    # Rehydrate into SelectionResult for consistent access
    if isinstance(selection_dict, SelectionResult):
        sr = selection_dict
    else:
        try:
            sr = SelectionResult.model_validate(selection_dict)
        except Exception:
            sr = SelectionResult()

    raw_directly = [s.model_dump() for s in sr.directly_supported]
    raw_pattern = [s.model_dump() for s in sr.pattern_inferred]
    raw_no_ev = [s.model_dump() for s in sr.no_evidence]

    # Fall back to converting verify_judgments directly if selection_result is empty
    if not raw_directly and not raw_pattern and not raw_no_ev:
        judgment_dicts = state.get("verify_judgments", [])
        if judgment_dicts:
            judgments = [
                CandidateJudgment(**j) if isinstance(j, dict) else j
                for j in judgment_dicts
            ]
            fallback = _judgments_to_selection_result(judgments)
            raw_directly = [s.model_dump() for s in fallback.directly_supported]
            raw_pattern = [s.model_dump() for s in fallback.pattern_inferred]
            raw_no_ev = [s.model_dump() for s in fallback.no_evidence]

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

    directly_supported = _dedup_vs_list(raw_directly)
    pattern_inferred = _dedup_vs_list(raw_pattern)
    no_evidence = raw_no_ev

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
