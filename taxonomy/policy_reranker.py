"""
Taxonomy policy reranker (Phase 4).

Applies label eligibility rules on top of evidence-based verify_judgments to
produce a reranked candidate list before finalize_selection (Pass 2).

This module answers: given that Pass 1 found evidence for stream X, is it
*appropriate* to label this idea card with stream X?

Evidence support (already handled by Phases 1-3) tells us *what* the card is
about. Label eligibility (this module) tells us *whether* that evidence is
strong enough to justify the label.

Pipeline:
  verify_candidates (Pass 1) → taxonomy_policy_rerank → finalize_selection (Pass 2)

Steps:
  1. Canonicalize all candidate names via taxonomy registry
  2. Join verify_judgments with candidate_evidence (fused scores, signals)
  3. Hard suppression: suppress streams flagged suppress_if_preferred when a
     preferred sibling is present and scoring higher
  4. Sibling dominance: demote streams within the same family if a clear
     dominant sibling exists
  5. Historical priors: use label_frequency priors to reweight adjacent_weak
     labels vs dominant_labels
  6. Policy signal adjustment: apply eligibility_boost from policy rules when
     text signals confirm or weaken a label
  7. Emit reranked/suppressed/decision lists

Output keys:
  taxonomy_reranked_candidates  - ordered list of surviving candidates (dicts)
  taxonomy_suppressed_candidates - suppressed candidates (dicts, for debug)
  taxonomy_decisions             - per-candidate decisions (dicts, for debug)
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

# Minimum text-signal matches required to confirm a label that has
# suppress_if_only_semantic=True (otherwise it is suppressed).
_SIGNAL_CONFIRM_MIN = 1

# Score delta required for sibling dominance: preferred must exceed weak by this much.
_SIBLING_DOMINANCE_DELTA = 0.12

# Score floor below which a candidate is considered "weak" regardless of policy.
_WEAK_SCORE_FLOOR = 0.35


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------

def _load_policy_rules(
    path: Optional[pathlib.Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load taxonomy_policy_rules.yaml. Returns {} on failure."""
    rules_path = path or _POLICY_RULES_PATH
    if not rules_path.exists():
        logger.warning("[policy_reranker] Policy rules not found: %s", rules_path)
        return {}
    try:
        with rules_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        streams = payload.get("streams") or {}
        if not isinstance(streams, dict):
            logger.warning("[policy_reranker] 'streams' is not a dict in %s", rules_path)
            return {}
        return streams
    except Exception as exc:
        logger.warning("[policy_reranker] Failed to load policy rules: %s", exc)
        return {}


def _load_historical_priors(
    path: Optional[pathlib.Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load historical_label_priors.yaml. Returns {} on failure."""
    priors_path = path or _PRIORS_PATH
    if not priors_path.exists():
        logger.warning("[policy_reranker] Historical priors not found: %s", priors_path)
        return {}
    try:
        with priors_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        families = payload.get("families") or {}
        if not isinstance(families, dict):
            logger.warning("[policy_reranker] 'families' is not a dict in %s", priors_path)
            return {}
        return families
    except Exception as exc:
        logger.warning("[policy_reranker] Failed to load historical priors: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Text signal helpers
# ---------------------------------------------------------------------------

def _count_signal_matches(lower_text: str, signals: List[str]) -> int:
    """Count how many signal phrases appear in lower_text."""
    return sum(1 for s in signals if s.lower() in lower_text)


def _compute_policy_adjustment(
    lower_text: str,
    rule: Dict[str, Any],
) -> float:
    """
    Compute eligibility score adjustment for a candidate using its policy rule.

    Returns:
      +eligibility_boost  if requires_any signals are found
      -eligibility_boost  if only weak_only_not_enough signals are found
       0.0                if no signals found or rule is absent
    """
    if not rule:
        return 0.0

    requires_any: List[str] = rule.get("requires_any") or []
    weak_only: List[str] = rule.get("weak_only_not_enough") or []
    boost = float(rule.get("eligibility_boost") or 0.0)

    positive_matches = _count_signal_matches(lower_text, requires_any)
    weak_matches = _count_signal_matches(lower_text, weak_only) if weak_only else 0

    if positive_matches >= 1:
        return boost  # confirmed by strong signal
    if weak_matches >= 1 and positive_matches == 0:
        return -abs(boost) if boost != 0.0 else -0.03  # weak-only penalty
    return 0.0


# ---------------------------------------------------------------------------
# Core reranker
# ---------------------------------------------------------------------------

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
    """
    Rerank verify_judgments using taxonomy policy rules and historical priors.

    Args:
        verify_judgments:        List of CandidateJudgment dicts from Pass 1.
        candidate_evidence:      List of CandidateEvidence dicts (for fused scores).
        taxonomy_registry:       TaxonomyRegistry instance (or None).
        taxonomy_policy_rules:   Pre-loaded policy rules dict (or None to load from file).
        historical_label_priors: Pre-loaded priors dict (or None to load from file).
        historical_value_stream_support:
                              Historical support rows from analog retrieval.
        bundle_patterns:       Bundle co-occurrence pattern rows.
        downstream_chains:     Downstream chain rows (upstream -> downstream).
        downstream_promoted_candidates:
                              Candidates promoted by downstream promoter node.
        lower_card_text:         Lowercased card text for signal matching.
        policy_rules_path:       Override path for taxonomy_policy_rules.yaml.
        priors_path:             Override path for historical_label_priors.yaml.

    Returns:
        Dict with keys:
          taxonomy_reranked_candidates   - list of surviving candidates (enriched dicts)
          taxonomy_suppressed_candidates - list of suppressed candidates
          taxonomy_decisions             - per-candidate decision records
    """
    # Load configs if not provided
    if taxonomy_policy_rules is None:
        taxonomy_policy_rules = _load_policy_rules(policy_rules_path)
    if historical_label_priors is None:
        historical_label_priors = _load_historical_priors(priors_path)

    # Build fused score lookup: canonical_name.lower() -> fused_score
    fused_lookup: Dict[str, float] = {}
    for ce in candidate_evidence or []:
        name = (ce.get("candidate_name") or ce.get("entity_name") or "").strip()
        if name:
            fused_lookup[name.lower()] = float(ce.get("fused_score") or 0.0)

    # Canonicalize judgments and build working list
    working: List[Dict[str, Any]] = []
    for jd in verify_judgments or []:
        raw_name = (jd.get("entity_name") or "").strip()
        if not raw_name:
            continue

        # Resolve to canonical name if registry available
        if taxonomy_registry is not None:
            canon_name = taxonomy_registry.canonicalize(raw_name)
        else:
            canon_name = raw_name

        fused_score = fused_lookup.get(canon_name.lower(), 0.0)

        working.append({
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
            "historical_support_score": 0.0,
            "bundle_support_score": 0.0,
            "downstream_support_score": 0.0,
            "suppressed": False,
            "suppression_reason": "",
            "eligibility_score": fused_score,  # will be updated below
        })

    # Build family membership maps from registry
    family_map: Dict[str, str] = {}  # canon_name -> family
    preferred_index: Dict[str, List[str]] = {}  # canon_name -> streams it dominates
    suppressible: Set[str] = set()

    if taxonomy_registry is not None:
        for stream in getattr(taxonomy_registry, "streams", []):
            family_map[stream.canonical_name] = stream.family
            if stream.preferred_over:
                preferred_index[stream.canonical_name] = list(stream.preferred_over)
            if stream.suppress_if_preferred:
                suppressible.add(stream.canonical_name)

    # Also pull family info from policy rules for streams not in registry
    for stream_name, rule in taxonomy_policy_rules.items():
        if stream_name not in family_map:
            fam = rule.get("family", "")
            if fam:
                family_map[stream_name] = fam

    # -----------------------------------------------------------------------
    # Step 1: Apply policy signal adjustments
    # -----------------------------------------------------------------------
    for entry in working:
        rule = taxonomy_policy_rules.get(entry["entity_name"]) or {}
        adj = _compute_policy_adjustment(lower_card_text, rule)
        name_l = _n(entry["entity_name"])
        hist_score = hist_score_map.get(name_l, 0.0)
        bundle_score = bundle_support_map.get(name_l, 0.0)
        downstream_score = downstream_score_map.get(name_l, 0.0)
        downstream_supported = (
            name_l in promoted_set
            or downstream_score >= 0.20
            or "downstream_promotion" in (entry.get("rationale", "") or "")
        )
        if downstream_supported and hist_score >= 0.25:
            downstream_score = max(downstream_score, 0.35)

        entry["historical_support_score"] = round(hist_score, 3)
        entry["bundle_support_score"] = round(bundle_score, 3)
        entry["downstream_support_score"] = round(downstream_score, 3)
        entry["historical_support_present"] = hist_score > 0.0
        entry["bundle_support_present"] = bundle_score > 0.0
        entry["downstream_support_present"] = downstream_supported or downstream_score > 0.0

        history_bonus = min(0.08, hist_score * 0.10)
        bundle_bonus = min(0.08, bundle_score * 0.12)
        downstream_bonus = min(0.10, downstream_score * 0.12)
        entry["policy_adjustment"] = adj
        entry["eligibility_score"] = round(
            max(0.0, entry["fused_score"] + adj + history_bonus + bundle_bonus + downstream_bonus), 3
        )

    # -----------------------------------------------------------------------
    # Step 2: Hard suppression — suppress_if_preferred + registry preferred_over
    # -----------------------------------------------------------------------
    # Build set of candidate names for quick membership check
    candidate_names = {e["entity_name"] for e in working}

    for preferred_stream, dominated_list in preferred_index.items():
        if preferred_stream not in candidate_names:
            continue
        # Find score of the preferred stream
        pref_entry = next(
            (e for e in working if e["entity_name"] == preferred_stream), None
        )
        if pref_entry is None:
            continue

        for dominated in dominated_list:
            if dominated not in candidate_names:
                continue
            if dominated not in suppressible:
                continue
            dom_entry = next(
                (e for e in working if e["entity_name"] == dominated), None
            )
            if dom_entry is None:
                continue

            # Suppress dominated stream if preferred scores significantly higher
            score_gap = pref_entry["eligibility_score"] - dom_entry["eligibility_score"]
            if score_gap >= _SIBLING_DOMINANCE_DELTA:
                dom_entry["suppressed"] = True
                dom_entry["suppression_reason"] = (
                    f"preferred_sibling:{preferred_stream}|gap:{score_gap:.3f}"
                )
                logger.debug(
                    "[policy_reranker] Suppressed %s — preferred by %s (gap=%.3f)",
                    dominated, preferred_stream, score_gap,
                )

    # -----------------------------------------------------------------------
    # Step 3: Semantic-only suppression
    # -----------------------------------------------------------------------
    for entry in working:
        if entry["suppressed"]:
            continue
        rule = taxonomy_policy_rules.get(entry["entity_name"]) or {}
        suppress_semantic = bool(rule.get("suppress_if_only_semantic", False))
        if not suppress_semantic:
            continue

        requires_any: List[str] = rule.get("requires_any") or []
        positive_matches = _count_signal_matches(lower_card_text, requires_any)

        # Downstream+historical justification can bypass semantic-only suppression.
        has_historical_justification = (
            entry.get("historical_support_score", 0.0) >= 0.25
            or entry.get("bundle_support_score", 0.0) >= 0.60
        )
        has_downstream_justification = bool(entry.get("downstream_support_present", False))

        if (
            positive_matches < _SIGNAL_CONFIRM_MIN
            and not (has_historical_justification and has_downstream_justification)
        ):
            # Check minimum_support_type requirement
            min_type = rule.get("minimum_support_type", "any")
            bucket = entry["bucket"]
            if min_type == "direct" and bucket != "directly_supported":
                entry["suppressed"] = True
                entry["suppression_reason"] = (
                    f"semantic_only:no_positive_signals|requires_direct_support"
                )
                logger.debug(
                    "[policy_reranker] Suppressed %s — semantic only, requires direct support",
                    entry["entity_name"],
                )
            elif min_type not in ("any",) and positive_matches < _SIGNAL_CONFIRM_MIN:
                entry["suppressed"] = True
                entry["suppression_reason"] = (
                    f"semantic_only:no_positive_signals|min_type={min_type}"
                )
                logger.debug(
                    "[policy_reranker] Suppressed %s — semantic only, no positive signals",
                    entry["entity_name"],
                )

    # -----------------------------------------------------------------------
    # Step 4: Historical prior adjustments for adjacent_weak_labels
    # -----------------------------------------------------------------------
    for entry in working:
        if entry["suppressed"]:
            continue
        family = family_map.get(entry["entity_name"], "")
        if not family:
            continue
        prior = historical_label_priors.get(family) or {}
        adjacent_weak: List[str] = prior.get("adjacent_weak_labels") or []

        if entry["entity_name"] in adjacent_weak:
            # Weak labels get a small penalty unless they have explicit positive signals
            rule = taxonomy_policy_rules.get(entry["entity_name"]) or {}
            requires_any: List[str] = rule.get("requires_any") or []
            pos_matches = _count_signal_matches(lower_card_text, requires_any)

            if pos_matches == 0 and entry["eligibility_score"] < _WEAK_SCORE_FLOOR:
                entry["policy_adjustment"] -= 0.05
                entry["eligibility_score"] = round(
                    max(0.0, entry["eligibility_score"] - 0.05), 3
                )
                logger.debug(
                    "[policy_reranker] Adjacent-weak penalty for %s (family=%s)",
                    entry["entity_name"], family,
                )

        # Strengthen dominant labels with historical priors when evidence exists.
        dominant_labels: List[str] = prior.get("dominant_labels") or []
        freq = float(prior.get("label_frequency") or 0.0)
        if (
            entry["entity_name"] in dominant_labels
            and (entry.get("historical_support_present") or entry.get("bundle_support_present"))
            and freq > 0.0
        ):
            boost = min(0.06, 0.03 * freq)
            entry["policy_adjustment"] += boost
            entry["eligibility_score"] = round(min(1.0, entry["eligibility_score"] + boost), 3)

    # -----------------------------------------------------------------------
    # Compile outputs
    # -----------------------------------------------------------------------
    reranked = []
    suppressed = []
    decisions = []

    for entry in working:
        decision = {
            "entity_name": entry["entity_name"],
            "original_name": entry["original_name"],
            "bucket": entry["bucket"],
            "confidence": entry["confidence"],
            "fused_score": entry["fused_score"],
            "eligibility_score": entry["eligibility_score"],
            "policy_adjustment": entry["policy_adjustment"],
            "suppressed": entry["suppressed"],
            "suppression_reason": entry["suppression_reason"],
            "historical_support_present": entry["historical_support_present"],
            "bundle_support_present": entry["bundle_support_present"],
            "downstream_support_present": entry["downstream_support_present"],
            "historical_support_score": entry["historical_support_score"],
            "bundle_support_score": entry["bundle_support_score"],
            "downstream_support_score": entry["downstream_support_score"],
        }
        decisions.append(decision)

        if entry["suppressed"]:
            suppressed.append({
                "entity_name": entry["entity_name"],
                "bucket": entry["bucket"],
                "eligibility_score": entry["eligibility_score"],
                "suppression_reason": entry["suppression_reason"],
                "historical_support_score": entry["historical_support_score"],
                "bundle_support_score": entry["bundle_support_score"],
                "downstream_support_score": entry["downstream_support_score"],
            })
        else:
            reranked.append({
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
                "historical_support_score": entry["historical_support_score"],
                "bundle_support_score": entry["bundle_support_score"],
                "downstream_support_score": entry["downstream_support_score"],
            })

    # Sort reranked: directly_supported first, then by eligibility_score desc
    bucket_order = {"directly_supported": 0, "pattern_inferred": 1, "no_evidence": 2}
    reranked.sort(
        key=lambda x: (
            bucket_order.get(x["bucket"], 9),
            -x["eligibility_score"],
        )
    )

    n_suppressed = len(suppressed)
    n_reranked = len(reranked)
    if n_suppressed > 0 or n_reranked > 0:
        logger.info(
            "[policy_reranker] Reranked %d candidates, suppressed %d | suppressed=%s",
            n_reranked, n_suppressed,
            [s["entity_name"] for s in suppressed],
        )

    return {
        "taxonomy_reranked_candidates": reranked,
        "taxonomy_suppressed_candidates": suppressed,
        "taxonomy_decisions": decisions,
    }
    def _n(name: str) -> str:
        return (name or "").strip().lower()

    hist_score_map: Dict[str, float] = {}
    for row in historical_value_stream_support or []:
        name = _n(row.get("entity_name") or row.get("candidate_name") or "")
        if not name:
            continue
        score = float(row.get("score") or 0.0)
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
        analog_count = float(row.get("analog_count") or 0.0)
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
