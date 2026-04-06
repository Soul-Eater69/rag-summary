"""
Summary-first retrieval: query FAISS summary index for analog tickets,
then optionally fetch raw chunks for top hits and KG candidates.

This replaces the old "retrieve all raw chunks first" approach with:
1. Summary FAISS search -> top analog historical tickets
2. KG candidate retrieval + value-stream definitions
3. Optional raw chunk lookup -> only for top shortlisted tickets

V5 additions:
- collect_value_stream_evidence now preserves capability_tags and
  operational_footprint from FAISS metadata for richer historical exploitation
- collect_attachment_candidates converts raw snippets into per-VS
  attachment source signals for CandidateEvidence
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from rag_summary.ingestion.faiss_indexer import search_summary_index, DEFAULT_INDEX_DIR
from rag_summary.ingestion.summary_generator import build_retrieval_text

logger = logging.getLogger(__name__)

def retrieve_analog_tickets(
    new_card_summary: Dict[str, Any],
    *,
    index_dir: str = DEFAULT_INDEX_DIR,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Query the FAISS summary index with the new card's retrieval text.

    Returns ranked analog historical tickets with metadata and scores.
    """
    query_text = build_retrieval_text(new_card_summary)
    
    if not query_text.strip():
        logger.warning("Empty retrieval text for new card summary, skipping FAISS search")
        return []
        
    results = search_summary_index(query_text, index_dir=index_dir, top_k=top_k)
    logger.info(
        "FAISS summary search returned %d analog tickets (top score=%.4f)",
        len(results),
        results[0]["score"] if results else 0.0,
    )
    return results

def retrieve_raw_evidence_for_tickets(
    ticket_ids: List[str],
    *,
    ticket_chunks_dir: str = "ticket_chunks",
    query_text: Optional[str] = None,
    max_chunks_per_ticket: int = 2,
    max_chunk_chars: int = 300,
) -> List[Dict[str, Any]]:
    """
    Late-stage raw chunk lookup for shortlisted tickets only.

    Fetches the best raw evidence snippets from ticket chunk files
    for verification, citation, and borderline candidate validation.

    Ranking: when ``query_text`` is provided, chunks are ranked by token
    overlap with the query (higher overlap = more relevant). Falls back to
    length-based ranking only when no query text is available.
    """
    chunks_path = pathlib.Path(ticket_chunks_dir)
    evidence: List[Dict[str, Any]] = []
    query_tokens = _tokenize(query_text) if query_text else set()

    for ticket_id in ticket_ids:
        ticket_dir = chunks_path / ticket_id
        raw_chunks = _load_raw_chunks(ticket_dir)

        if not raw_chunks:
            continue

        if query_tokens:
            # Rank by token overlap with the query text (more relevant = higher score)
            ranked = sorted(
                raw_chunks,
                key=lambda c: _overlap_score(c.get("text") or c.get("content") or "", query_tokens),
                reverse=True,
            )
        else:
            # Fallback: prefer longer chunks as a rough substance proxy
            ranked = sorted(raw_chunks, key=lambda c: len(c.get("text", "")), reverse=True)

        for chunk in ranked[:max_chunks_per_ticket]:
            text = (chunk.get("text") or chunk.get("content") or "").strip()
            if not text:
                continue
            evidence.append({
                "ticket_id": ticket_id,
                "chunk_id": chunk.get("chunk_id", ""),
                "snippet": text[:max_chunk_chars],
                "provenance": chunk.get("provenance", ""),
            })

    logger.info("Retrieved %d raw evidence snippets for %d tickets", len(evidence), len(ticket_ids))
    return evidence

def retrieve_kg_candidates(
    cleaned_text: str,
    *,
    top_k: int = 20,
    allowed_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve value-stream candidates from the KG index (Azure AI Search).

    Reuses the existing retrieval pipeline's KG search.
    """
    from src.pipelines.value_stream.retrieval_pipeline import (
        retrieve_kg_candidates as _retrieve_kg,
    )

    candidates = _retrieve_kg(cleaned_text, top_k=top_k, allowed_names=allowed_names)
    logger.info("Retrieved %d KG candidates", len(candidates))
    return candidates

def collect_value_stream_evidence(
    analog_tickets: List[Dict[str, Any]],
    *,
    ticket_chunks_dir: str = "ticket_chunks",
) -> List[Dict[str, Any]]:
    """
    Collect value-stream label evidence from analog tickets' ground truth maps.

    Returns aggregated VS support with counts and source ticket IDs.
    """
    chunks_path = pathlib.Path(ticket_chunks_dir)
    vs_support: Dict[str, Dict[str, Any]] = {}

    for ticket in analog_tickets:
        ticket_id = ticket.get("ticket_id", "")
        if not ticket_id:
            continue

        # From FAISS metadata
        labels = ticket.get("value_stream_labels", [])

        # Also try loading from 08 file directly
        if not labels:
            vs_map_path = chunks_path / ticket_id / "08_valuestream_map.json"
            if vs_map_path.exists():
                try:
                    with open(vs_map_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    labels = data.get("valueStreamNames", []) or []
                except Exception:
                    pass

        score = float(ticket.get("score", 0.0))
        # V5: prefer canonical functions; fall back to legacy field
        ticket_functions = list(
            ticket.get("direct_functions_canonical")
            or ticket.get("direct_functions")
            or []
        )
        ticket_actors = list(ticket.get("actors") or [])
        # V5: richer historical metadata
        ticket_cap_tags = list(ticket.get("capability_tags") or [])
        ticket_footprint = list(ticket.get("operational_footprint") or [])
        ticket_support_type = dict(ticket.get("stream_support_type") or {})
        ticket_supporting_evidence = list(ticket.get("supporting_evidence") or [])

        for label in labels:
            label = label.strip()
            if not label:
                continue
            key = label.lower()
            if key not in vs_support:
                vs_support[key] = {
                    "entity_name": label,
                    "support_count": 0,
                    "best_score": 0.0,
                    "supporting_ticket_ids": [],
                    "supporting_functions": [],
                    "supporting_actors": [],
                    # V5: aggregated capability / footprint / evidence
                    "capability_tags": [],
                    "operational_footprint": [],
                    "supporting_evidence": [],
                    "stream_support_types": [],
                }

            entry = vs_support[key]
            entry["support_count"] += 1
            entry["best_score"] = max(entry["best_score"], score)
            if ticket_id not in entry["supporting_ticket_ids"]:
                entry["supporting_ticket_ids"].append(ticket_id)
            for fn in ticket_functions:
                if fn and fn not in entry["supporting_functions"]:
                    entry["supporting_functions"].append(fn)
            for actor in ticket_actors:
                if actor and actor not in entry["supporting_actors"]:
                    entry["supporting_actors"].append(actor)
            # V5: accumulate capability signals from each supporting analog
            for tag in ticket_cap_tags:
                if tag and tag not in entry["capability_tags"]:
                    entry["capability_tags"].append(tag)
            for fp in ticket_footprint:
                if fp and fp not in entry["operational_footprint"]:
                    entry["operational_footprint"].append(fp)
            for ev in ticket_supporting_evidence:
                if ev and ev not in entry["supporting_evidence"]:
                    entry["supporting_evidence"].append(ev)
            # Record how this specific analog classified the stream
            if label in ticket_support_type:
                entry["stream_support_types"].append(ticket_support_type[label])

    result = sorted(vs_support.values(), key=lambda x: (-x["support_count"], -x["best_score"]))
    logger.info("Collected %d value-stream evidence entries from %d analog tickets", len(result), len(analog_tickets))
    return result


def collect_attachment_candidates(
    raw_evidence: List[Dict[str, Any]],
    analog_tickets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build attachment-source candidates for CandidateEvidence from raw evidence.

    Strategy: for each analog ticket that contributed raw snippets, its
    mapped value streams receive an attachment source score proportional to
    the analog's similarity score. This is a sound proxy because:
    - we are not inventing new VS names
    - attachment text comes from the analog's own artifact files
    - if the analog was relevant enough to retrieve, its raw text is
      genuine supporting evidence for the streams it was mapped to

    Returns candidates shaped for build_candidate_evidence's chunk/attachment path.
    """
    if not raw_evidence:
        return []

    # Build ticket_id -> snippet list map
    ticket_snippets: Dict[str, List[str]] = {}
    for ev in raw_evidence:
        tid = ev.get("ticket_id", "")
        snippet = ev.get("snippet", "")
        if tid and snippet:
            ticket_snippets.setdefault(tid, []).append(snippet)

    # Build ticket_id -> {score, value_stream_labels} from analogs
    analog_by_id: Dict[str, Dict[str, Any]] = {
        t["ticket_id"]: t for t in analog_tickets if t.get("ticket_id")
    }

    candidates: List[Dict[str, Any]] = []
    seen: set = set()

    for ticket_id, snippets in ticket_snippets.items():
        analog = analog_by_id.get(ticket_id)
        if not analog:
            continue
        analog_score = float(analog.get("score", 0.0))
        # Attachment score: attenuate the analog similarity score
        attachment_score = round(analog_score * 0.75, 4)

        for label in analog.get("value_stream_labels", []):
            label = label.strip()
            if not label or label in seen:
                continue
            seen.add(label)
            candidates.append({
                "entity_name": label,
                "score": attachment_score,
                "source": "attachment",
                "sub_source": "attachment_proxy",   # V6: marks as analog-derived
                "snippets": snippets[:2],
            })

    return candidates

def enrich_historical_candidates(
    vs_support: List[Dict[str, Any]],
    *,
    new_card_summary: Optional[Dict[str, Any]] = None,
    analog_summaries: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Enrich vs_support entries with support-type-weighted scores and evidence phrases.

    V6 additions over V5:

    1. Score calibration by dominant support type (unchanged from V5):
         - "direct"     → ×1.00
         - "downstream" → ×0.80
         - "pattern"    → ×0.60

    2. V6: Capability overlap blending.
       If new_card_summary and analog_summaries are provided, compute the max
       Jaccard capability-tag overlap across all analogs that supported this VS.
       Final score = 0.70 × type_calibrated + 0.30 × capability_overlap.
       This gives more weight to analogs whose capability profile closely matches
       the new card, independent of FAISS similarity.

    3. Evidence phrase enrichment (unchanged from V5 + overlap phrase when strong).

    Returns a copy of vs_support with added keys:
      - score: final blended score
      - dominant_support_type: str
      - capability_overlap_score: float (max overlap across supporting analogs)
      - evidence_phrases: List[str]
    """
    from rag_summary.retrieval.history_patterns import compute_capability_overlap

    _TYPE_WEIGHT = {"direct": 1.00, "downstream": 0.80, "pattern": 0.60}

    # Build lookup: ticket_id → analog dict (for capability overlap)
    analog_by_id: Dict[str, Dict[str, Any]] = {}
    if analog_summaries:
        for a in analog_summaries:
            tid = a.get("ticket_id") or a.get("id") or ""
            if tid:
                analog_by_id[tid] = a

    result = []
    for entry in vs_support:
        support_types = entry.get("stream_support_types") or []
        base_score = float(entry.get("best_score") or 0.0)

        # Dominant support type
        type_counts: Dict[str, int] = {}
        for t in support_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        dominant_type = max(type_counts, key=lambda k: type_counts[k]) if type_counts else "pattern"

        weight = _TYPE_WEIGHT.get(dominant_type, 0.70)
        calibrated_score = round(min(1.0, base_score * weight), 4)

        # V6: capability overlap across supporting analogs
        cap_overlap = 0.0
        if new_card_summary and analog_by_id:
            for tid in entry.get("supporting_ticket_ids", []):
                analog = analog_by_id.get(tid)
                if analog:
                    overlap = compute_capability_overlap(new_card_summary, analog)
                    cap_overlap = max(cap_overlap, overlap)

        # Blend type-calibrated score with capability overlap
        if cap_overlap > 0.0:
            final_score = round(0.70 * calibrated_score + 0.30 * cap_overlap, 4)
        else:
            final_score = calibrated_score

        # Evidence phrases
        evidence_phrases: List[str] = []
        for fp in (entry.get("operational_footprint") or [])[:4]:
            if fp and fp.strip():
                evidence_phrases.append(f"footprint: {fp}")
        for ev in (entry.get("supporting_evidence") or [])[:3]:
            if ev and ev.strip():
                evidence_phrases.append(f"evidence: {ev}")
        if cap_overlap >= 0.40:
            evidence_phrases.append(
                f"capability-overlap: {cap_overlap:.0%} shared capability tags with supporting analogs"
            )

        result.append({
            **entry,
            "score": final_score,
            "dominant_support_type": dominant_type,
            "capability_overlap_score": cap_overlap,
            "evidence_phrases": evidence_phrases,
        })

    return result


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Return a set of lower-cased word tokens (3+ chars) for overlap scoring."""
    return {w.lower() for w in (text or "").split() if len(w) >= 3}

def _overlap_score(chunk_text: str, query_tokens: set) -> float:
    """Fraction of query tokens that appear in the chunk text."""
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenize(chunk_text)
    return len(query_tokens & chunk_tokens) / len(query_tokens)

def _load_raw_chunks(ticket_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Load raw chunks from a ticket directory."""
    chunks: List[Dict[str, Any]] = []

    for candidate in ["07_chunks.json", "03_retrieval_views.json", "04_chunks.json"]:
        path = ticket_dir / candidate
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Current format: {"chunks": [{"id": "...", "content": "..."}, ...]}
            if candidate == "07_chunks.json" and isinstance(data, dict):
                raw_list = data.get("chunks", [])
                if isinstance(raw_list, list):
                    for item in raw_list:
                        if not isinstance(item, dict):
                            continue
                        text = (item.get("text") or item.get("content") or "").strip()
                        if not text:
                            continue
                        chunks.append({
                            "text": text,
                            "chunk_id": item.get("chunk_id") or item.get("id") or "",
                            "provenance": item.get("provenance") or "07_chunks.json",
                        })
                continue

            if isinstance(data, list):
                chunks.extend(data)
            elif isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, list):
                        chunks.extend(val)
                    elif isinstance(val, str) and len(val) > 20:
                        chunks.append({"text": val, "chunk_id": key})
        except Exception:
            continue

    return chunks
