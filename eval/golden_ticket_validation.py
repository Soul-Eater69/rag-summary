"""
Single-ticket golden validation helper for taxonomy policy hardening.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_summary.eval.canonicalize_predictions import (
    canonicalize_predictions,
    extract_predicted_names,
)
from rag_summary.eval.eval_taxonomy_metrics import compute_exact_metrics, compute_family_metrics
from rag_summary.pipeline import run_summary_rag_pipeline


def _normalize_names(names: List[str], registry: Optional[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for name in names or []:
        canonical = registry.canonicalize(name) if registry is not None else name
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def validate_single_ticket(
    *,
    ticket_text: str,
    gt_labels: List[str],
    registry: Optional[Any] = None,
    debug_output_dir: Optional[str] = None,
    **pipeline_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run one ticket end-to-end, canonicalize, compare to GT, and expose debug-focused diagnostics.
    """
    result = run_summary_rag_pipeline(
        ticket_text,
        debug_output_dir=debug_output_dir,
        taxonomy_registry=registry,
        **pipeline_kwargs,
    )
    canonical = canonicalize_predictions(result, registry=registry)
    predicted_labels = extract_predicted_names(canonical, include_pattern=True)
    canonical_gt = _normalize_names(gt_labels, registry=registry)

    exact = compute_exact_metrics(predicted_labels, canonical_gt)
    family = compute_family_metrics(predicted_labels, canonical_gt, registry=registry)

    suppressed = canonical.get("taxonomy_suppressed", [])
    verify_names = {
        (j.get("entity_name") or "").strip()
        for j in result.get("verify_judgments", [])
        if (j.get("entity_name") or "").strip()
    }
    promoted_names = {
        (c.get("entity_name") or "").strip()
        for c in result.get("downstream_promoted_candidates", [])
        if (c.get("entity_name") or "").strip()
    }

    false_negatives = exact.get("false_negatives", [])
    miss_categories: Dict[str, str] = {}
    for label in false_negatives:
        if label in suppressed:
            miss_categories[label] = "taxonomy_suppression"
        elif label in promoted_names and label not in verify_names:
            miss_categories[label] = "downstream_promotion_miss"
        elif label in verify_names:
            miss_categories[label] = "finalize_or_selection_miss"
        elif label in canonical.get("unknown_names", []):
            miss_categories[label] = "alias_or_canonicalization_mismatch"
        else:
            miss_categories[label] = "upstream_candidate_miss"

    report = {
        "predicted_labels": predicted_labels,
        "suppressed_labels": suppressed,
        "gt_labels": canonical_gt,
        "false_positives": exact.get("false_positives", []),
        "false_negatives": false_negatives,
        "family_matches": family.get("matched_families", []),
        "policy_decisions": result.get("taxonomy_decisions", []),
        "miss_categories": miss_categories,
        "exact_metrics": exact,
        "family_metrics": family,
        "canonical_prediction_labels": predicted_labels,
        "canonical_gt_labels": canonical_gt,
    }
    return report
