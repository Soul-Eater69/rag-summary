"""
Canonicalize prediction output for eval comparison.

Takes raw pipeline output (directly_supported / pattern_inferred) and
maps all value stream names to their canonical form using the taxonomy
registry.  Produces a normalized prediction dict ready for metric computation.

Usage:
    from rag_summary.eval.canonicalize_predictions import canonicalize_predictions

    # Single prediction
    canonical = canonicalize_predictions(pipeline_output, registry)

    # Batch (list of pipeline outputs)
    results = [canonicalize_predictions(o, registry) for o in outputs]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def canonicalize_predictions(
    pipeline_output: Dict[str, Any],
    registry: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Normalize all value stream names in pipeline_output to canonical form.

    Args:
        pipeline_output: Final state dict from run_prediction_graph().
        registry:        TaxonomyRegistry instance.  When None, names are
                         returned unchanged.

    Returns:
        Dict with keys:
          directly_supported  - list of {"entity_name": canonical, "confidence": float, ...}
          pattern_inferred    - list of {"entity_name": canonical, ...}
          no_evidence         - list of {"entity_name": canonical, ...}
          selected_value_streams - union (direct + pattern) with canonical names
          taxonomy_suppressed    - list of suppressed names (for debug)
          unknown_names          - list of names not found in registry
    """

    def _canon(name: str) -> str:
        if registry is None:
            return name
        return registry.canonicalize(name)

    def _normalize_list(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for item in items or []:
            raw_name = item.get("entity_name") or item.get("name") or ""
            canon_name = _canon(raw_name.strip())
            out.append({**item, "entity_name": canon_name})
        return out

    directly = _normalize_list(pipeline_output.get("directly_supported") or [])
    pattern = _normalize_list(pipeline_output.get("pattern_inferred") or [])
    no_ev = _normalize_list(pipeline_output.get("no_evidence") or [])
    selected = _normalize_list(pipeline_output.get("selected_value_streams") or [])

    # Suppressed candidates from Phase 4 taxonomy reranker
    suppressed_raw = pipeline_output.get("taxonomy_suppressed_candidates") or []
    suppressed_names = [
        _canon((s.get("entity_name") or "").strip())
        for s in suppressed_raw
    ]

    # Detect names that could not be resolved (still differ after canonicalization)
    all_vs = registry.canonical_label_map if registry is not None else {}
    unknown: List[str] = []

    for item in (directly + pattern + no_ev):
        name = item.get("entity_name", "")
        if registry is not None and name.lower() not in all_vs:
            unknown.append(name)

    if unknown:
        logger.debug(
            "[canonicalize_predictions] %d unknown names: %s",
            len(unknown), unknown,
        )

    return {
        "directly_supported": directly,
        "pattern_inferred": pattern,
        "no_evidence": no_ev,
        "selected_value_streams": selected,
        "taxonomy_suppressed": suppressed_names,
        "unknown_names": unknown,
    }


def extract_predicted_names(
    canonical_output: Dict[str, Any],
    *,
    include_pattern: bool = True,
) -> List[str]:
    """
    Return a flat list of predicted value stream names from canonicalized output.

    Args:
        canonical_output: Output of canonicalize_predictions().
        include_pattern:  If True, include pattern_inferred names.

    Returns:
        Deduplicated list of entity_name strings.
    """
    seen = set()
    names: List[str] = []

    for item in canonical_output.get("directly_supported") or []:
        n = item.get("entity_name", "")
        if n and n not in seen:
            seen.add(n)
            names.append(n)

    if include_pattern:
        for item in canonical_output.get("pattern_inferred") or []:
            n = item.get("entity_name", "")
            if n and n not in seen:
                seen.add(n)
                names.append(n)

    return names
