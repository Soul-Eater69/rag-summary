"""
CandidateEvidence builder (V5 architecture).

Builds the core runtime artifact: a CandidateEvidence object per candidate
value stream, with multi-source provenance, scores, snippets, and support
type classification.

This is the most important runtime object in the V5 architecture -- it makes
the system debuggable, auditable, and calibratable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from core.text import normalize_for_search

logger = logging.getLogger(__name__)

# Source keys used throughout the pipeline
SOURCE_CHUNK = "chunk"
SOURCE_SUMMARY = "summary"
SOURCE_ATTACHMENT = "attachment"
SOURCE_THEME = "theme"
SOURCE_KG = "kg"
SOURCE_HISTORICAL = "historical"
SOURCE_CAPABILITY = "capability"

ALL_SOURCES = [
    SOURCE_CHUNK, SOURCE_SUMMARY, SOURCE_ATTACHMENT,
    SOURCE_THEME, SOURCE_KG, SOURCE_HISTORICAL, SOURCE_CAPABILITY,
]


def _norm(name: str) -> str:
    return normalize_for_search((name or "").strip())


def build_candidate_evidence(
    *,
    kg_candidates: List[Dict[str, Any]],
    historical_candidates: List[Dict[str, Any]],
    capability_candidates: List[Dict[str, Any]],
    chunk_candidates: Optional[List[Dict[str, Any]]] = None,
    attachment_candidates: Optional[List[Dict[str, Any]]] = None,
    theme_candidates: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build CandidateEvidence objects from multiple sources.

    Each source provides candidates as dicts with at least:
      - entity_name: str
      - score: float (0..1 preferred)
      - entity_id: str (optional)
      - description: str (optional)
      - snippets: list of {source, snippet} (optional)

    Returns a list of CandidateEvidence dicts, one per unique candidate,
    with merged provenance from all sources.
    """
    # Accumulate by normalized name
    evidence_map: Dict[str, Dict[str, Any]] = {}

    def _ensure_entry(name: str, entity_id: str = "", description: str = "") -> Dict[str, Any]:
        key = _norm(name)
        if not key:
            return {}
        if key not in evidence_map:
            evidence_map[key] = {
                "candidate_id": entity_id or "",
                "candidate_name": name.strip(),
                "source_scores": {s: 0.0 for s in ALL_SOURCES},
                "evidence_sources": [],
                "evidence_snippets": [],
                "fused_score": 0.0,
                "support_confidence": 0.0,
                "source_diversity_count": 0,
                "support_type": "none",
                "contradictions": [],
            }
        entry = evidence_map[key]
        if entity_id and not entry["candidate_id"]:
            entry["candidate_id"] = entity_id
        if description and not entry.get("description"):
            entry["description"] = description
        return entry

    def _add_source(
        candidates: List[Dict[str, Any]],
        source_key: str,
    ) -> None:
        for cand in candidates or []:
            name = (cand.get("entity_name") or cand.get("name") or "").strip()
            if not name:
                continue
            entry = _ensure_entry(
                name,
                entity_id=cand.get("entity_id", ""),
                description=(cand.get("description") or "")[:300],
            )
            if not entry:
                continue

            score = float(cand.get("score") or cand.get("best_score") or 0.0)
            score = max(0.0, min(1.0, score))
            entry["source_scores"][source_key] = max(
                entry["source_scores"][source_key], score
            )

            if source_key not in entry["evidence_sources"]:
                entry["evidence_sources"].append(source_key)

            # Collect snippets
            for snippet in cand.get("snippets", []):
                entry["evidence_snippets"].append({
                    "source": source_key,
                    "snippet": snippet if isinstance(snippet, str) else snippet.get("snippet", ""),
                })

            # Also treat supporting_evidence as snippets
            for ev in cand.get("supporting_evidence", []):
                if isinstance(ev, str) and ev.strip():
                    entry["evidence_snippets"].append({
                        "source": source_key,
                        "snippet": ev,
                    })

    # Ingest from each source
    _add_source(kg_candidates, SOURCE_KG)
    _add_source(historical_candidates, SOURCE_HISTORICAL)
    _add_source(capability_candidates, SOURCE_CAPABILITY)
    _add_source(chunk_candidates or [], SOURCE_CHUNK)
    _add_source(attachment_candidates or [], SOURCE_ATTACHMENT)
    _add_source(theme_candidates or [], SOURCE_THEME)

    # Compute diversity and classify support type
    result = list(evidence_map.values())
    for entry in result:
        active_sources = [
            s for s in ALL_SOURCES if entry["source_scores"].get(s, 0.0) > 0.0
        ]
        entry["source_diversity_count"] = len(active_sources)
        entry["evidence_sources"] = active_sources
        entry["support_type"] = _classify_support_type(entry)

    return result


def _classify_support_type(entry: Dict[str, Any]) -> str:
    """
    Classify a candidate's support type based on which sources contributed.

    Returns one of: "direct", "pattern", "mixed", "none"
    """
    scores = entry.get("source_scores", {})

    direct_sources = {SOURCE_CHUNK, SOURCE_SUMMARY, SOURCE_ATTACHMENT, SOURCE_KG}
    pattern_sources = {SOURCE_HISTORICAL, SOURCE_CAPABILITY}

    has_direct = any(scores.get(s, 0.0) > 0.0 for s in direct_sources)
    has_pattern = any(scores.get(s, 0.0) > 0.0 for s in pattern_sources)

    if has_direct and has_pattern:
        return "mixed"
    elif has_direct:
        return "direct"
    elif has_pattern:
        return "pattern"
    return "none"
