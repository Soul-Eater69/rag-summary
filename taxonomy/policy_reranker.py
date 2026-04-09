"""
Taxonomy policy reranker (Phase 4).
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

_POLICY_RULES_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "config" / "taxonomy_policy_rules.yaml"
)
_PRIORS_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "config" / "historical_label_priors.yaml"
)

_SIGNAL_CONFIRM_MIN = 1
_SIBLING_DOMINANCE_DELTA = 0.12
_WEAK_SCORE_FLOOR = 0.35


def _load_policy_rules(path: Optional[pathlib.Path] = None) -> Dict[str, Dict[str, Any]]:
    rules_path = path or _POLICY_RULES_PATH
    if not rules_path.exists():
        logger.warning("[policy_reranker] Policy rules not found: %s", rules_path)
        return {}
    try:
        with rules_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        streams = payload.get("streams") or {}
        return streams if isinstance(streams, dict) else {}
    except Exception as exc:
        logger.warning("[policy_reranker] Failed to load policy rules: %s", exc)
        return {}


def _load_historical_priors(path: Optional[pathlib.Path] = None) -> Dict[str, Dict[str, Any]]:
    priors_path = path or _PRIORS_PATH
    if not priors_path.exists():
        logger.warning("[policy_reranker] Historical priors not found: %s", priors_path)
        return {}
    try:
        with priors_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        families = payload.get("families") or {}
        return families if isinstance(families, dict) else {}
    except Exception as exc:
        logger.warning("[policy_reranker] Failed to load historical priors: %s", exc)
        return {}


def _count_signal_matches(lower_text: str, signals: List[str]) -> int:
    return sum(1 for s in signals if s and s.lower() in lower_text)


def _compute_policy_adjustment(lower_text: str, rule: Dict[str, Any]) -> float:
    if not rule:
        return 0.0

    requires_any: List[str] = rule.get("requires_any") or []
    weak_only: List[str] = rule.get("weak_only_not_enough") or []
    boost = float(rule.get("eligibility_boost") or 0.0)

    positive_matches = _count_signal_matches(lower_text, requires_any)
    weak_matches = _count_signal_matches(lower_text, weak_only) if weak_only else 0

    if positive_matches >= 1:
        return boost
    if weak_matches >= 1 and positive_matches == 0:
        return -abs(boost) if boost != 0.0 else -0.03
    return 0.0


def _n(name: str) -> str:
    return (name or "").strip().lower()


def _build_historical_maps(
    historical_value_stream_support: Optional[List[Dict[str, Any]]],
    bundle_patterns: Optional[List[Dict[str, Any]]],
    downstream_chains: Optional[List[Dict[str, Any]]],
    downstream_promoted_candidates: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    hist_score_map: Dict[str, float] = {}
    for row in historical_value_stream_support or []:
        name = _n(row.get("entity_name") or row.get("candidate_name") or "")
        if not name:
            continue
        score = float(row.get("score") or row.get("support_score") or 0.0)
        hist_score_map[name] = max(hist_score_map.get(name, 0.0), score)

    bundle_support_map: Dict[str, float] = {}
    for row in bundle_patterns or []:
        fraction = float(row.get("co_occurrence_fraction") or 0.0)
        primary = _n(row.get("primary_vs") or "")
        bundled = _n(row.get("bundled_vs") or "")
        if primary:
            bundle_support_map[primary] = max(bundle_support_map.get(primary, 0.0), fraction)
        if bundled:
            bundle_support_map[bundled] = max(bundle_support_map.get(bundled, 0.0), fraction)

    downstream_score_map: Dict[str, float] = {}
    for row in downstream_chains or []:
        up = _n(row.get("upstream_vs") or "")
        down = _n(row.get("downstream_vs") or "")
        analog_count = float(row.get("analog_count") or row.get("count") or 0.0)
        score = min(1.0, analog_count / 5.0) if analog_count > 0 else 0.0
        if down:
            downstream_score_map[down] = max(downstream_score_map.get(down, 0.0), score)
        if up:
            downstream_score_map[up] = max(downstream_score_map.get(up, 0.0), score * 0.5)

    promoted_set = {
        _n(c.get("entity_name") or c.get("candidate_name") or "")
        for c in (downstream_promoted_candidates or [])
        if _n(c.get("entity_name") or c.get("candidate_name") or "")
    }

    return {
        "hist_score_map": hist_score_map,
        "bundle_support_map": bundle_support_map,
        "downstream_score_map": downstream_score_map,
        "promoted_set": promoted_set,
    }


def rerank_candidates_by_taxonomy_policy(
    *,
    verify_judgments: List[Dict[str, Any]],
    candidate_evidence: List[Dict[str, Any]],
    taxonomy_registry: Any,
    taxonomy_policy_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    historical_label_priors: Optional[Dict[str, Dict[str, Any]]] = None,
    historical_value_stream_support: Optional[List[Dict[str, Any]]] = None,
    bundle_patterns: Optional[List[Dict[str, Any]]] = None,
    downstream_chains: Optional[List[Dict[str, Any]]] = None,
    downstream_promoted_candidates: Optional[List[Dict[str, Any]]] = None,
    lower_card_text: str = "",
    policy_rules_path: Optional[pathlib.Path] = None,
    priors_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    if taxonomy_policy_rules is None:
        taxonomy_policy_rules = _load_policy_rules(policy_rules_path)
    if historical_label_priors is None:
        historical_label_priors = _load_historical_priors(priors_path)

    maps = _build_historical_maps(
        historical_value_stream_support,
        bundle_patterns,
        downstream_chains,
        downstream_promoted_candidates,
    )
    hist_score_map = maps["hist_score_map"]
    bundle_support_map = maps["bundle_support_map"]
    downstream_score_map = maps["downstream_score_map"]
    promoted_set = maps["promoted_set"]

    fused_lookup: Dict[str, float] = {}
    for ce in candidate_evidence or []:
        name = (ce.get("candidate_name") or ce.get("entity_name") or "").strip()
        if name:
            fused_lookup[_n(name)] = float(ce.get("fused_score") or 0.0)

    working: List[Dict[str, Any]] = []
    for jd in verify_judgments or []:
        raw_name = (jd.get("entity_name") or "").strip()
        if not raw_name:
            continue
        canon_name = taxonomy_registry.canonicalize(raw_name) if taxonomy_registry is not None else raw_name
        name_l = _n(canon_name)
        fused_score = fused_lookup.get(name_l, 0.0)
        working.append(
            {
                "entity_name": canon_name,
                "original_name": raw_name,
                "bucket": jd.get("bucket", "no_evidence"),
                "confidence": float(jd.get("confidence") or 0.0),
                "rationale": jd.get("rationale") or "",
                "fused_score": fused_score,
                "policy_adjustment": 0.0,
                "historical_support_present": False,
                "bundle_support_present": False,
                "downstream_support_present": False,
                "downstream_promoted": False,
                "historical_prior_family_match": False,
                "historical_support_score": 0.0,
                "bundle_support_score": 0.0,
                "downstream_support_score": 0.0,
                "suppressed": False,
                "demoted": False,
                "suppression_reason": "",
                "decision_reason": "kept",
                "historical_contribution": False,
                "bundle_downstream_contribution": False,
                "sibling_dominance_contribution": False,
                "text_signal_confirmed": False,
                "eligibility_score": fused_score,
            }
        )

    family_map: Dict[str, str] = {}
    preferred_index: Dict[str, List[str]] = {}
    suppressible: Set[str] = set()
    if taxonomy_registry is not None:
        for stream in getattr(taxonomy_registry, "streams", []):
            family_map[stream.canonical_name] = stream.family
            if stream.preferred_over:
                preferred_index[stream.canonical_name] = list(stream.preferred_over)
            if stream.suppress_if_preferred:
                suppressible.add(stream.canonical_name)

    for stream_name, rule in (taxonomy_policy_rules or {}).items():
        if stream_name not in family_map and rule.get("family"):
            family_map[stream_name] = rule.get("family")
        prefer_over = rule.get("prefer_over") or []
        if prefer_over:
            preferred_index.setdefault(stream_name, [])
            preferred_index[stream_name].extend(prefer_over)
            for dominated in prefer_over:
                if bool((taxonomy_policy_rules.get(dominated) or {}).get("suppress_if_preferred", True)):
                    suppressible.add(dominated)

    for entry in working:
        rule = taxonomy_policy_rules.get(entry["entity_name"]) or {}
        requires_any: List[str] = rule.get("requires_any") or []
        positive_matches = _count_signal_matches(lower_card_text, requires_any)
        adj = _compute_policy_adjustment(lower_card_text, rule)

        name_l = _n(entry["entity_name"])
        hist_score = hist_score_map.get(name_l, 0.0)
        bundle_score = bundle_support_map.get(name_l, 0.0)
        downstream_score = downstream_score_map.get(name_l, 0.0)
        downstream_promoted = name_l in promoted_set
        downstream_supported = downstream_promoted or downstream_score >= 0.20 or "downstream_promotion" in entry["rationale"]

        family = family_map.get(entry["entity_name"], "")
        prior = historical_label_priors.get(family) or {}
        downstream_labels = set(prior.get("downstream_labels") or [])
        dominant_labels = set(prior.get("dominant_labels") or [])

        historical_prior_family_match = bool(family and prior)
        if entry["entity_name"] in downstream_labels and (bundle_score >= 0.5 or downstream_supported):
            hist_score = max(hist_score, 0.30)

        if downstream_supported and hist_score >= 0.25:
            downstream_score = max(downstream_score, 0.35)

        history_bonus = min(0.10, hist_score * 0.12)
        bundle_bonus = min(0.10, bundle_score * 0.15)
        downstream_bonus = min(0.12, downstream_score * 0.14)

        if historical_prior_family_match and entry["entity_name"] in dominant_labels and hist_score > 0.0:
            history_bonus = min(0.12, history_bonus + 0.03)

        entry["historical_support_score"] = round(hist_score, 3)
        entry["bundle_support_score"] = round(bundle_score, 3)
        entry["downstream_support_score"] = round(downstream_score, 3)
        entry["historical_support_present"] = hist_score > 0.0
        entry["bundle_support_present"] = bundle_score > 0.0
        entry["downstream_support_present"] = downstream_supported or downstream_score > 0.0
        entry["downstream_promoted"] = downstream_promoted
        entry["historical_prior_family_match"] = historical_prior_family_match
        entry["text_signal_confirmed"] = positive_matches >= _SIGNAL_CONFIRM_MIN
        entry["historical_contribution"] = history_bonus > 0.0
        entry["bundle_downstream_contribution"] = (bundle_bonus + downstream_bonus) > 0.0
        entry["policy_adjustment"] = adj
        entry["eligibility_score"] = round(max(0.0, fused_score + adj + history_bonus + bundle_bonus + downstream_bonus), 3)

    candidate_names = {e["entity_name"] for e in working}

    for preferred_stream, dominated_list in preferred_index.items():
        if preferred_stream not in candidate_names:
            continue
        pref_entry = next((e for e in working if e["entity_name"] == preferred_stream), None)
        if pref_entry is None:
            continue

        for dominated in dominated_list:
            if dominated not in candidate_names or dominated not in suppressible:
                continue
            dom_entry = next((e for e in working if e["entity_name"] == dominated), None)
            if dom_entry is None:
                continue

            score_gap = pref_entry["eligibility_score"] - dom_entry["eligibility_score"]
            if score_gap >= _SIBLING_DOMINANCE_DELTA:
                dom_entry["suppressed"] = True
                dom_entry["demoted"] = True
                dom_entry["decision_reason"] = "suppressed"
                dom_entry["sibling_dominance_contribution"] = True
                dom_entry["suppression_reason"] = f"preferred_sibling:{preferred_stream}|gap:{score_gap:.3f}"

    for entry in working:
        if entry["suppressed"]:
            continue
        rule = taxonomy_policy_rules.get(entry["entity_name"]) or {}
        if not bool(rule.get("suppress_if_only_semantic", False)):
            continue

        requires_any: List[str] = rule.get("requires_any") or []
        positive_matches = _count_signal_matches(lower_card_text, requires_any)

        has_historical_justification = (
            entry.get("historical_support_score", 0.0) >= 0.25
            or entry.get("bundle_support_score", 0.0) >= 0.60
        )
        has_downstream_justification = bool(entry.get("downstream_support_present", False))

        if positive_matches < _SIGNAL_CONFIRM_MIN and not (has_historical_justification and has_downstream_justification):
            min_type = rule.get("minimum_support_type", "any")
            if min_type == "direct" and entry["bucket"] != "directly_supported":
                entry["suppressed"] = True
                entry["decision_reason"] = "suppressed"
                entry["suppression_reason"] = "semantic_only:no_positive_signals|requires_direct_support"
            elif min_type not in ("any",) and positive_matches < _SIGNAL_CONFIRM_MIN:
                entry["suppressed"] = True
                entry["decision_reason"] = "suppressed"
                entry["suppression_reason"] = f"semantic_only:no_positive_signals|min_type={min_type}"

    for entry in working:
        if entry["suppressed"]:
            continue
        family = family_map.get(entry["entity_name"], "")
        prior = historical_label_priors.get(family) or {}

        adjacent_weak: List[str] = prior.get("adjacent_weak_labels") or []
        if entry["entity_name"] in adjacent_weak:
            rule = taxonomy_policy_rules.get(entry["entity_name"]) or {}
            requires_any: List[str] = rule.get("requires_any") or []
            pos_matches = _count_signal_matches(lower_card_text, requires_any)
            if pos_matches == 0 and entry["eligibility_score"] < _WEAK_SCORE_FLOOR and entry["historical_support_score"] < 0.20:
                entry["policy_adjustment"] -= 0.05
                entry["demoted"] = True
                entry["decision_reason"] = "demoted"
                entry["eligibility_score"] = round(max(0.0, entry["eligibility_score"] - 0.05), 3)

        if entry["historical_prior_family_match"] and not entry["historical_support_present"] and entry["text_signal_confirmed"]:
            # Penalize broad semantically plausible labels lacking historical support.
            entry["policy_adjustment"] -= 0.03
            entry["demoted"] = True
            entry["decision_reason"] = "demoted"
            entry["eligibility_score"] = round(max(0.0, entry["eligibility_score"] - 0.03), 3)

        dominant_labels: List[str] = prior.get("dominant_labels") or []
        freq = float(prior.get("label_frequency") or 0.0)
        if entry["entity_name"] in dominant_labels and (entry["historical_support_present"] or entry["bundle_support_present"]) and freq > 0.0:
            boost = min(0.06, 0.03 * freq)
            entry["policy_adjustment"] += boost
            entry["historical_contribution"] = True
            entry["eligibility_score"] = round(min(1.0, entry["eligibility_score"] + boost), 3)

    reranked: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    for entry in working:
        if entry["decision_reason"] == "kept" and not entry["suppressed"] and not entry["demoted"]:
            entry["decision_reason"] = "kept"

        decision = {
            "entity_name": entry["entity_name"],
            "original_name": entry["original_name"],
            "bucket": entry["bucket"],
            "confidence": entry["confidence"],
            "fused_score": entry["fused_score"],
            "eligibility_score": entry["eligibility_score"],
            "policy_adjustment": entry["policy_adjustment"],
            "suppressed": entry["suppressed"],
            "demoted": entry["demoted"],
            "decision_reason": entry["decision_reason"],
            "suppression_reason": entry["suppression_reason"],
            "historical_support_present": entry["historical_support_present"],
            "bundle_support_present": entry["bundle_support_present"],
            "downstream_support_present": entry["downstream_support_present"],
            "downstream_promoted": entry["downstream_promoted"],
            "historical_prior_family_match": entry["historical_prior_family_match"],
            "historical_support_score": entry["historical_support_score"],
            "bundle_support_score": entry["bundle_support_score"],
            "downstream_support_score": entry["downstream_support_score"],
            "historical_contribution": entry["historical_contribution"],
            "bundle_downstream_contribution": entry["bundle_downstream_contribution"],
            "sibling_dominance_contribution": entry["sibling_dominance_contribution"],
            "text_signal_confirmed": entry["text_signal_confirmed"],
        }
        decisions.append(decision)

        payload = {
            "entity_name": entry["entity_name"],
            "bucket": entry["bucket"],
            "confidence": entry["confidence"],
            "rationale": entry["rationale"],
            "fused_score": entry["fused_score"],
            "eligibility_score": entry["eligibility_score"],
            "policy_adjustment": entry["policy_adjustment"],
            "historical_support_present": entry["historical_support_present"],
            "bundle_support_present": entry["bundle_support_present"],
            "downstream_support_present": entry["downstream_support_present"],
            "downstream_promoted": entry["downstream_promoted"],
            "historical_prior_family_match": entry["historical_prior_family_match"],
            "historical_support_score": entry["historical_support_score"],
            "bundle_support_score": entry["bundle_support_score"],
            "downstream_support_score": entry["downstream_support_score"],
            "decision_reason": entry["decision_reason"],
        }

        if entry["suppressed"]:
            payload["suppression_reason"] = entry["suppression_reason"]
            suppressed.append(payload)
        else:
            reranked.append(payload)

    bucket_order = {"directly_supported": 0, "pattern_inferred": 1, "no_evidence": 2}
    reranked.sort(key=lambda x: (bucket_order.get(x["bucket"], 9), -x["eligibility_score"]))

    logger.info(
        "[policy_reranker] Reranked %d candidates, suppressed %d | suppressed=%s",
        len(reranked), len(suppressed), [s["entity_name"] for s in suppressed],
    )

    return {
        "taxonomy_reranked_candidates": reranked,
        "taxonomy_suppressed_candidates": suppressed,
        "taxonomy_decisions": decisions,
    }
