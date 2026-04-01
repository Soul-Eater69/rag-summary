"""
Summary-first retrieval: query FAISS summary index for analog tickets,
then optionally fetch raw chunks for top hits and KG candidates.

This replaces the old "retrieve all raw chunks first" approach with:
  1. Summary FAISS search → top analog historical tickets
  2. KG candidate retrieval + value-stream definitions
  3. Optional raw chunk lookup → only for top shortlisted tickets
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from summary_rag.ingestion.faiss_indexer import search_summary_index, DEFAULT_INDEX_DIR
from summary_rag.ingestion.summary_generator import build_retrieval_text

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
    max_chunks_per_ticket: int = 2,
    max_chunk_chars: int = 300,
) -> List[Dict[str, Any]]:
    """
    Late-stage raw chunk lookup for shortlisted tickets only.

    Fetches the best raw evidence snippets from ticket chunk files
    for verification, citation, and borderline candidate validation.
    """
    chunks_path = pathlib.Path(ticket_chunks_dir)
    evidence: List[Dict[str, Any]] = []

    for ticket_id in ticket_ids:
        ticket_dir = chunks_path / ticket_id
        raw_chunks = _load_raw_chunks(ticket_dir)

        # Take top N chunks by length (longer = more substance)
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
                    with open(vs_map_path, encoding="utf-8") as f:
                        data = json.load(f)
                    labels = data.get("valueStreamNames", []) or []
                except Exception:
                    pass

        score = float(ticket.get("score", 0.0))
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
                }
            entry = vs_support[key]
            entry["support_count"] += 1
            entry["best_score"] = max(entry["best_score"], score)
            if ticket_id not in entry["supporting_ticket_ids"]:
                entry["supporting_ticket_ids"].append(ticket_id)

    result = sorted(vs_support.values(), key=lambda x: (-x["support_count"], -x["best_score"]))
    logger.info("Collected %d value-stream evidence entries from %d analog tickets", len(result), len(analog_tickets))
    return result


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_raw_chunks(ticket_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Load raw chunks from a ticket directory."""
    chunks: List[Dict[str, Any]] = []

    for candidate in ["03_retrieval_views.json", "04_chunks.json"]:
        path = ticket_dir / candidate
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
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