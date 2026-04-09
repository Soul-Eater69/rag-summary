"""
Downstream candidate promotion (Phase 3).

Promotes value stream candidates that are not directly evidenced in card text
but are strongly implied by historical bundle patterns, downstream chains, or
high-scoring prerequisite streams.

Called by node_promote_downstream_candidates (between parse_attachments and
build_evidence). The output is merged into the candidate pool as "pattern"
support_type candidates before evidence building and fusion.

Rule evaluation:
  For each rule in downstream_promotion_rules.yaml:
    1. Check if any requires_any stream clears the threshold via:
       a) historical_value_stream_support score >= historical_score_min
       b) bundle_patterns co-occurrence fraction >= historical_bundle_min
       c) downstream_chains entry (when downstream_chain_trigger=true)
    2. If satisfied, emit a promoted candidate for each stream in promotes[]:
       - score = min(max_score, best_prerequisite_score * decay_factor)
       - support_type = "pattern"
       - sub_source = "downstream_promotion"
    3. If the target stream already has historical support >= max_score,
       skip promotion (it doesn't need help).

Score calculation:
  promoted_score = min(rule.max_score, best_trigger_score * 0.65)

  decay_factor=0.65 ensures promoted candidates sit below directly-evidenced
  ones and don't crowd out streams with real textual support.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

_RULES_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "downstream_promotion_rules.yaml"

_DECAY_FACTOR = 0.65  # promoted score = trigger_score * decay


def _load_promotion_rules(path: Optional[pathlib.Path] = None) -> List[Dict[str, Any]]:
    """Load downstream promotion rules from YAML. Returns empty list on failure."""
    rules_path = path or _RULES_PATH
    if not rules_path.exists():
        logger.warning("[downstream_promoter] Rules file not found: %s", rules_path)
        return []
    try:
        with rules_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        rules = payload.get("rules") or []
        if not isinstance(rules, list):
            logger.warning("[downstream_promoter] 'rules' key is not a list in %s", rules_path)
            return []
        return rules
    except Exception as exc:
        logger.warning("[downstream_promoter] Failed to load rules: %s", exc)
        return []


def _build_historical_score_map(
    historical_value_stream_support: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Build a normalized name -> max score map from historical VS support.
    Uses lower-stripped names for case-insensitive lookup.
    """
    score_map: Dict[str, float] = {}
    for entry in historical_value_stream_support or []:
        name = (entry.get("entity_name") or entry.get("name") or "").strip()
        if not name:
            continue
        score = float(entry.get("score") or entry.get("best_score") or 0.0)
        key = name.lower()
        score_map[key] = max(score_map.get(key, 0.0), score)
    return score_map


def _build_bundle_score_map(
    bundle_patterns: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Build a normalized name -> max co_occurrence_fraction map from bundle patterns.
    Considers both primary_vs and bundled_vs in each pattern.
    """
    bundle_map: Dict[str, float] = {}
    for bp in bundle_patterns or []:
        fraction = float(bp.get("co_occurrence_fraction") or 0.0)
        for field in ("primary_vs", "bundled_vs"):
            name = (bp.get(field) or "").strip()
            if name:
                key = name.lower()
                bundle_map[key] = max(bundle_map.get(key, 0.0), fraction)
    return bundle_map


def _build_downstream_set(
    downstream_chains: List[Dict[str, Any]],
) -> Set[str]:
    """
    Build a set of (upstream_vs.lower(), downstream_vs.lower()) pairs from
    downstream chains. Also adds all upstream_vs names for quick lookup.
    """
    upstream_names: Set[str] = set()
    for dc in downstream_chains or []:
        up = (dc.get("upstream_vs") or "").strip().lower()
        if up:
            upstream_names.add(up)
    return upstream_names


def promote_downstream_candidates(
    *,
    historical_value_stream_support: List[Dict[str, Any]],
    bundle_patterns: List[Dict[str, Any]],
    downstream_chains: List[Dict[str, Any]],
    allowed_names: Optional[Set[str]] = None,
    rules_path: Optional[pathlib.Path] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate downstream promotion rules and return promoted candidate dicts.

    Args:
        historical_value_stream_support: VS support entries from collect_vs_evidence.
        bundle_patterns: VS pair co-occurrence entries from collect_vs_evidence.
        downstream_chains: Downstream activation chain entries.
        allowed_names: If provided, only emit candidates whose name is in this set.
        rules_path: Override path to downstream_promotion_rules.yaml (for testing).

    Returns:
        List of candidate dicts with:
          entity_name, score, source="historical", sub_source="downstream_promotion",
          support_type="pattern", match_reason, rule_name
    """
    rules = _load_promotion_rules(rules_path)
    if not rules:
        return []

    hist_scores = _build_historical_score_map(historical_value_stream_support)
    bundle_scores = _build_bundle_score_map(bundle_patterns)
    downstream_upstreams = _build_downstream_set(downstream_chains)

    # Also build an existing historical score map (to skip streams already well-covered)
    existing_hist = {k: v for k, v in hist_scores.items()}

    promoted: Dict[str, Dict[str, Any]] = {}  # target_name.lower() -> best candidate

    for rule in rules:
        rule_name = rule.get("name", "unnamed")
        requires_any: List[str] = rule.get("requires_any") or []
        hist_min = float(rule.get("historical_score_min") or 0.0)
        bundle_min = float(rule.get("historical_bundle_min") or 0.0)
        chain_trigger = bool(rule.get("downstream_chain_trigger", False))
        promotes: List[str] = rule.get("promotes") or []
        max_score = float(rule.get("max_score") or 0.65)

        if not requires_any or not promotes:
            continue

        # Find the best trigger score across requires_any streams
        best_trigger_score = 0.0
        trigger_name = ""
        trigger_source = ""

        for req in requires_any:
            key = req.lower()

            # Check historical support
            hist_score = hist_scores.get(key, 0.0)
            if hist_score >= hist_min and hist_score > best_trigger_score:
                best_trigger_score = hist_score
                trigger_name = req
                trigger_source = f"historical_score:{hist_score:.3f}"

            # Check bundle co-occurrence
            bund_score = bundle_scores.get(key, 0.0)
            if bund_score >= bundle_min:
                # Convert bundle fraction to a score signal (not a raw score)
                bund_as_score = round(bund_score * 0.80, 3)  # fraction → attenuated score
                if bund_as_score > best_trigger_score:
                    best_trigger_score = bund_as_score
                    trigger_name = req
                    trigger_source = f"bundle_fraction:{bund_score:.2f}"

            # Check downstream chain trigger
            if chain_trigger and key in downstream_upstreams:
                chain_score = max(0.45, hist_scores.get(key, 0.45))
                if chain_score > best_trigger_score:
                    best_trigger_score = chain_score
                    trigger_name = req
                    trigger_source = f"downstream_chain"

        if best_trigger_score <= 0.0:
            continue

        # Rule fired — emit promoted candidates
        promoted_score = round(min(max_score, best_trigger_score * _DECAY_FACTOR), 3)

        for target in promotes:
            target_key = target.lower()

            # Skip if the target already has strong historical support on its own
            existing_score = existing_hist.get(target_key, 0.0)
            if existing_score >= max_score:
                logger.debug(
                    "[downstream_promoter] Skipping rule=%s target=%s: "
                    "already has historical score %.3f >= max_score %.3f",
                    rule_name, target, existing_score, max_score,
                )
                continue

            if allowed_names is not None and target not in allowed_names:
                continue

            match_reason = (
                f"downstream_rule:{rule_name}|"
                f"trigger={trigger_name}|"
                f"trigger_source={trigger_source}"
            )

            # Max-pool if the same target is promoted by multiple rules
            existing_promoted = promoted.get(target_key)
            if existing_promoted is None or promoted_score > existing_promoted["score"]:
                promoted[target_key] = {
                    "entity_name": target,
                    "score": promoted_score,
                    "source": "historical",
                    "sub_source": "downstream_promotion",
                    "support_type": "pattern",
                    "match_reason": match_reason,
                    "rule_name": rule_name,
                    "trigger_name": trigger_name,
                    "trigger_score": best_trigger_score,
                }
                logger.debug(
                    "[downstream_promoter] Promoted %s (score=%.3f) via rule=%s trigger=%s",
                    target, promoted_score, rule_name, trigger_name,
                )

    result = list(promoted.values())
    if result:
        logger.info(
            "[downstream_promoter] %d downstream candidates promoted: %s",
            len(result),
            [c["entity_name"] for c in result],
        )
    return result
