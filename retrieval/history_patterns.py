"""
Historical footprint pattern detection (V6 architecture).

Provides three analytical functions that reason over analog tickets and VS
support data to surface co-occurrence and downstream activation patterns.

These patterns are then injected as evidence snippets (sub_source=bundle_pattern
or sub_source=downstream_chain) into CandidateEvidence during node_build_evidence.

Functions:
  detect_bundle_patterns   — VS pairs co-occurring across ≥ threshold of analogs
  detect_downstream_chains — VS entries consistently activated downstream
  compute_capability_overlap — Jaccard similarity between new card and an analog

All functions are pure (no I/O, no LLM calls) and operate on plain dicts from state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data shapes (plain dicts to avoid Pydantic overhead in hot path)
# ---------------------------------------------------------------------------

def _bundle_pattern(
    primary_vs: str,
    bundled_vs: str,
    co_occurrence_count: int,
    co_occurrence_fraction: float,
    supporting_analog_ids: List[str],
) -> Dict[str, Any]:
    return {
        "primary_vs": primary_vs,
        "bundled_vs": bundled_vs,
        "co_occurrence_count": co_occurrence_count,
        "co_occurrence_fraction": co_occurrence_fraction,
        "supporting_analog_ids": supporting_analog_ids[:5],
    }


def _downstream_chain(
    upstream_vs: str,
    downstream_vs: str,
    analog_count: int,
    avg_downstream_score: float,
    supporting_analog_ids: List[str],
) -> Dict[str, Any]:
    return {
        "upstream_vs": upstream_vs,
        "downstream_vs": downstream_vs,
        "analog_count": analog_count,
        "avg_downstream_score": avg_downstream_score,
        "supporting_analog_ids": supporting_analog_ids[:5],
    }


# ---------------------------------------------------------------------------
# Bundle pattern detection
# ---------------------------------------------------------------------------

def detect_bundle_patterns(
    analog_tickets: List[Dict[str, Any]],
    *,
    min_co_occurrence_fraction: float = 0.60,
    min_analog_count: int = 2,
    allowed_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Find VS pairs that co-occur in ≥ min_co_occurrence_fraction of analogs.

    Uses analog_tickets directly (each has value_stream_labels) so no change
    to collect_value_stream_evidence() is required.

    Example: if 4 of 5 analogs have both "Manage Enrollment" and "Configure Products",
    that pair is a bundle with fraction=0.80. The system uses this to emit a
    bundle_pattern evidence snippet on the bundled VS candidate.

    Returns list of bundle pattern dicts sorted by fraction descending.
    """
    if len(analog_tickets) < min_analog_count:
        return []

    allowed_set: Optional[Set[str]] = set(allowed_names) if allowed_names else None

    # Build per-analog VS label sets
    analog_vs_sets: List[Set[str]] = []
    analog_ids: List[str] = []
    for ticket in analog_tickets:
        tid = ticket.get("ticket_id", "")
        labels = ticket.get("value_stream_labels", [])
        if not labels:
            continue
        vs_set = set()
        for lbl in labels:
            lbl = lbl.strip()
            if lbl and (allowed_set is None or lbl in allowed_set):
                vs_set.add(lbl)
        if vs_set:
            analog_vs_sets.append(vs_set)
            analog_ids.append(tid)

    if len(analog_vs_sets) < min_analog_count:
        return []

    # Count per-VS occurrences and pair co-occurrences
    vs_count: Dict[str, int] = {}
    pair_tids: Dict[tuple, List[str]] = {}

    for vs_set, tid in zip(analog_vs_sets, analog_ids):
        for vs in vs_set:
            vs_count[vs] = vs_count.get(vs, 0) + 1
        sorted_vs = sorted(vs_set)
        for i, a in enumerate(sorted_vs):
            for b in sorted_vs[i + 1:]:
                key = (a, b)
                pair_tids.setdefault(key, []).append(tid)

    # Emit bundle patterns
    patterns: List[Dict[str, Any]] = []
    seen_directed: Set[tuple] = set()

    for (a, b), tids in pair_tids.items():
        count = len(tids)
        if count < min_analog_count:
            continue

        # Emit both directions: (a→b bundled) and (b→a bundled)
        for primary, bundled in [(a, b), (b, a)]:
            anchor_count = vs_count.get(primary, 1)
            fraction = round(count / max(anchor_count, 1), 4)
            if fraction >= min_co_occurrence_fraction:
                directed_key = (primary, bundled)
                if directed_key not in seen_directed:
                    seen_directed.add(directed_key)
                    patterns.append(_bundle_pattern(
                        primary_vs=primary,
                        bundled_vs=bundled,
                        co_occurrence_count=count,
                        co_occurrence_fraction=fraction,
                        supporting_analog_ids=tids,
                    ))

    patterns.sort(key=lambda p: -p["co_occurrence_fraction"])
    logger.info("[detect_bundle_patterns] found %d bundle patterns from %d analogs",
                len(patterns), len(analog_vs_sets))
    return patterns


# ---------------------------------------------------------------------------
# Downstream chain detection
# ---------------------------------------------------------------------------

def detect_downstream_chains(
    vs_support: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Find VS entries consistently activated downstream of direct VS activations.

    Uses stream_support_types from collect_value_stream_evidence (already populated).
    A VS with dominant_support_type="downstream" is paired against the direct VS
    entries that share the most analog overlap.

    Returns a list of downstream chain dicts.
    """
    chains: List[Dict[str, Any]] = []

    # Find direct and downstream VS entries
    direct_entries = [
        e for e in vs_support
        if _dominant_type(e.get("stream_support_types", [])) == "direct"
    ]
    downstream_entries = [
        e for e in vs_support
        if _dominant_type(e.get("stream_support_types", [])) == "downstream"
    ]

    if not downstream_entries:
        return []

    for down_entry in downstream_entries:
        down_name = down_entry.get("entity_name", "")
        if not down_name:
            continue
        down_tids = set(down_entry.get("supporting_ticket_ids", []))
        down_score = float(down_entry.get("best_score", 0.0)) * 0.80

        # Find the best-overlapping direct VS
        best_overlap = 0
        best_upstream = ""
        best_shared_tids: List[str] = []

        for dir_entry in direct_entries:
            dir_name = dir_entry.get("entity_name", "")
            dir_tids = set(dir_entry.get("supporting_ticket_ids", []))
            overlap = down_tids & dir_tids
            if len(overlap) >= 2 and len(overlap) > best_overlap:
                best_overlap = len(overlap)
                best_upstream = dir_name
                best_shared_tids = sorted(overlap)

        if best_upstream:
            chains.append(_downstream_chain(
                upstream_vs=best_upstream,
                downstream_vs=down_name,
                analog_count=best_overlap,
                avg_downstream_score=round(down_score, 4),
                supporting_analog_ids=best_shared_tids,
            ))

    logger.info("[detect_downstream_chains] found %d downstream chains", len(chains))
    return chains


# ---------------------------------------------------------------------------
# Capability overlap scoring
# ---------------------------------------------------------------------------

def compute_capability_overlap(
    new_card_summary: Dict[str, Any],
    analog: Dict[str, Any],
) -> float:
    """
    Compute Jaccard-based capability overlap between a new card summary and an analog.

    Blends:
      - capability_tags Jaccard (weight 0.70)
      - direct_functions_canonical Jaccard (weight 0.30)

    Both inputs are plain dicts (from state — not Pydantic objects).
    Returns float 0.0–1.0.
    """
    # Capability tags
    new_caps: Set[str] = {t.lower() for t in (new_card_summary.get("capability_tags") or [])}
    analog_caps: Set[str] = {t.lower() for t in (analog.get("capability_tags") or [])}
    union_caps = new_caps | analog_caps
    cap_jaccard = len(new_caps & analog_caps) / len(union_caps) if union_caps else 0.0

    # Canonical functions
    new_funcs: Set[str] = {f.lower() for f in (new_card_summary.get("direct_functions_canonical") or [])}
    analog_funcs: Set[str] = {f.lower() for f in (analog.get("direct_functions_canonical") or [])}
    union_funcs = new_funcs | analog_funcs
    func_jaccard = len(new_funcs & analog_funcs) / len(union_funcs) if union_funcs else 0.0

    result = round(min(1.0, 0.70 * cap_jaccard + 0.30 * func_jaccard), 4)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dominant_type(types: List[str]) -> str:
    """Return the most common support type from a list, defaulting to 'pattern'."""
    if not types:
        return "pattern"
    counts: Dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    return max(counts, key=lambda k: counts[k])
