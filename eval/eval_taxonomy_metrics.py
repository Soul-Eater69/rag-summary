"""
Taxonomy-aware evaluation metrics (Phase 4).

Computes precision/recall/F1 against ground-truth labels using canonical
taxonomy names.  Supports family-level aggregation and family-aware partial
credit (streams in the same family count as partial matches).

Usage:
    from rag_summary.eval.eval_taxonomy_metrics import (
        compute_exact_metrics,
        compute_family_metrics,
        evaluate_batch,
    )

    # Single card
    metrics = compute_exact_metrics(predicted=["Order to Cash"], ground_truth=["Order to Cash"])

    # Family-aware
    fmetrics = compute_family_metrics(predicted, ground_truth, registry)

    # Batch
    report = evaluate_batch(predictions_list, ground_truth_list, registry)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exact match metrics
# ---------------------------------------------------------------------------

def compute_exact_metrics(
    predicted: List[str],
    ground_truth: List[str],
) -> Dict[str, Any]:
    """
    Compute standard precision/recall/F1 using exact name matching.

    Returns:
        Dict with: precision, recall, f1, tp, fp, fn
    """
    pred_set = set(predicted)
    gt_set = set(ground_truth)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted": sorted(predicted),
        "ground_truth": sorted(ground_truth),
        "correct": sorted(pred_set & gt_set),
        "false_positives": sorted(pred_set - gt_set),
        "false_negatives": sorted(gt_set - pred_set),
    }


# ---------------------------------------------------------------------------
# Family-aware metrics
# ---------------------------------------------------------------------------

def _get_family(name: str, registry: Any) -> str:
    """Return the family label for a stream name, or '' if unknown."""
    if registry is None:
        return ""
    stream = registry.get_stream(name)
    return stream.family if stream else ""


def compute_family_metrics(
    predicted: List[str],
    ground_truth: List[str],
    registry: Optional[Any] = None,
    *,
    partial_credit: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute family-aware precision/recall/F1.

    A predicted stream counts as a partial match (partial_credit) if it
    belongs to the same family as a ground-truth stream, even if the exact
    name differs.

    Args:
        predicted:      Predicted canonical names.
        ground_truth:   Ground-truth canonical names.
        registry:       TaxonomyRegistry instance.
        partial_credit: Score weight for family-level matches (default 0.5).

    Returns:
        Dict with: exact_*, family_*, per_family_breakdown
    """
    exact = compute_exact_metrics(predicted, ground_truth)

    # Build family memberships
    pred_families = {_get_family(n, registry) for n in predicted if n}
    gt_families = {_get_family(n, registry) for n in ground_truth if n}

    # Family-level TP: predicted family matches a GT family
    matched_families = pred_families & gt_families
    unmatched_pred_families = pred_families - gt_families
    unmatched_gt_families = gt_families - pred_families

    # Partial credit for family matches (excluding exact matches)
    exact_names = set(predicted) & set(ground_truth)
    family_only_pred = [
        n for n in predicted
        if n not in exact_names and _get_family(n, registry) in gt_families
    ]
    family_only_gt = [
        n for n in ground_truth
        if n not in exact_names and _get_family(n, registry) in pred_families
    ]

    family_precision_score = (
        exact["tp"] + partial_credit * len(family_only_pred)
    ) / (len(predicted) if predicted else 1)

    family_recall_score = (
        exact["tp"] + partial_credit * len(family_only_gt)
    ) / (len(ground_truth) if ground_truth else 1)

    family_f1 = (
        2 * family_precision_score * family_recall_score
        / (family_precision_score + family_recall_score)
        if (family_precision_score + family_recall_score) > 0
        else 0.0
    )

    # Per-family breakdown
    all_families = pred_families | gt_families
    per_family: Dict[str, Dict[str, Any]] = {}
    for fam in sorted(all_families):
        if not fam:
            continue
        pred_in_fam = [n for n in predicted if _get_family(n, registry) == fam]
        gt_in_fam = [n for n in ground_truth if _get_family(n, registry) == fam]
        per_family[fam] = compute_exact_metrics(pred_in_fam, gt_in_fam)

    return {
        **{f"exact_{k}": v for k, v in exact.items()
           if k in ("precision", "recall", "f1", "tp", "fp", "fn")},
        "family_precision": round(family_precision_score, 4),
        "family_recall": round(family_recall_score, 4),
        "family_f1": round(family_f1, 4),
        "matched_families": sorted(matched_families - {""}),
        "unmatched_pred_families": sorted(unmatched_pred_families - {""}),
        "unmatched_gt_families": sorted(unmatched_gt_families - {""}),
        "family_only_pred": family_only_pred,
        "family_only_gt": family_only_gt,
        "per_family_breakdown": per_family,
    }


# ---------------------------------------------------------------------------
# Suppression metrics
# ---------------------------------------------------------------------------

def compute_suppression_metrics(
    suppressed: List[str],
    ground_truth: List[str],
) -> Dict[str, Any]:
    """
    Evaluate quality of taxonomy suppression decisions.

    A suppression is:
      - correct_suppression: suppressed a stream NOT in ground truth (good)
      - incorrect_suppression: suppressed a stream that IS in ground truth (bad)

    Returns:
        Dict with: correct_suppressions, incorrect_suppressions,
                   suppression_precision (correct / total_suppressed)
    """
    supp_set = set(suppressed)
    gt_set = set(ground_truth)

    correct = supp_set - gt_set       # suppressed things not in GT (correct)
    incorrect = supp_set & gt_set     # suppressed things that ARE in GT (wrong)

    precision = len(correct) / len(supp_set) if supp_set else 1.0

    return {
        "total_suppressed": len(supp_set),
        "correct_suppressions": sorted(correct),
        "incorrect_suppressions": sorted(incorrect),
        "suppression_precision": round(precision, 4),
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    predictions: List[Dict[str, Any]],
    ground_truths: List[List[str]],
    registry: Optional[Any] = None,
    *,
    include_pattern: bool = True,
) -> Dict[str, Any]:
    """
    Run exact + family metrics across a batch of (prediction, ground_truth) pairs.

    Args:
        predictions:    List of canonicalize_predictions() outputs.
        ground_truths:  Parallel list of ground-truth name lists.
        registry:       TaxonomyRegistry instance.
        include_pattern: Include pattern_inferred in predicted set.

    Returns:
        Dict with:
          per_card:    list of per-card metric dicts
          macro_avg:   macro-averaged metrics across cards
          micro_avg:   micro-averaged metrics (aggregate TP/FP/FN)
          family_agg:  family-level aggregated metrics
    """
    from rag_summary.eval.canonicalize_predictions import extract_predicted_names

    assert len(predictions) == len(ground_truths), (
        f"predictions ({len(predictions)}) and ground_truths ({len(ground_truths)}) "
        f"must have equal length"
    )

    per_card = []
    total_tp = total_fp = total_fn = 0
    family_correct: Dict[str, int] = defaultdict(int)
    family_total_pred: Dict[str, int] = defaultdict(int)
    family_total_gt: Dict[str, int] = defaultdict(int)

    for i, (pred_output, gt) in enumerate(zip(predictions, ground_truths)):
        pred_names = extract_predicted_names(pred_output, include_pattern=include_pattern)
        suppressed = pred_output.get("taxonomy_suppressed") or []
        exact = compute_exact_metrics(pred_names, gt)
        family = compute_family_metrics(pred_names, gt, registry)
        supp = compute_suppression_metrics(suppressed, gt)

        per_card.append({
            "card_index": i,
            "exact": exact,
            "family": family,
            "suppression": supp,
        })

        total_tp += exact["tp"]
        total_fp += exact["fp"]
        total_fn += exact["fn"]

        # Accumulate per-family counts
        for fam, fam_metrics in family.get("per_family_breakdown", {}).items():
            family_correct[fam] += fam_metrics["tp"]
            family_total_pred[fam] += fam_metrics["tp"] + fam_metrics["fp"]
            family_total_gt[fam] += fam_metrics["tp"] + fam_metrics["fn"]

    n = len(per_card)

    def _macro_avg(key: str, subkey: str) -> float:
        vals = [c[key][subkey] for c in per_card if subkey in c.get(key, {})]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    macro_precision = _macro_avg("exact", "precision")
    macro_recall = _macro_avg("exact", "recall")
    macro_f1 = _macro_avg("exact", "f1")
    macro_family_f1 = _macro_avg("family", "family_f1")

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    # Per-family aggregate precision/recall/F1
    family_agg: Dict[str, Dict[str, float]] = {}
    for fam in sorted(set(family_correct) | set(family_total_pred)):
        tp_f = family_correct.get(fam, 0)
        pred_f = family_total_pred.get(fam, 0)
        gt_f = family_total_gt.get(fam, 0)
        fp_f = pred_f - tp_f
        fn_f = gt_f - tp_f
        p = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
        r = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        family_agg[fam] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": tp_f, "fp": fp_f, "fn": fn_f,
        }

    return {
        "n_cards": n,
        "per_card": per_card,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "family_f1": macro_family_f1,
        },
        "micro_avg": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "family_agg": family_agg,
    }
