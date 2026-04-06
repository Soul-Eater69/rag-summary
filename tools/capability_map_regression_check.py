"""
Capability map regression checker (V6).

Compares a new capability_map.yaml against a baseline (prior version)
and reports any regressions: missing VS coverage, dropped cues, removed clusters.

Use this before deploying a rebuilt capability map to catch accidental
degradations in VS coverage or cue quality.

Exit code: 0 (no regressions), 1 (regressions found), 2 (error)

Usage:
    python -m summary_rag.tools.capability_map_regression_check \\
        --baseline config/capability_map.baseline.yaml \\
        --new config/capability_map.yaml \\
        [--fail-on-regression] \\
        [--min-coverage-pct 90.0]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Set, Tuple

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_map(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        sys.exit(2)


def _get_capabilities(cap_map: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return cap_map.get("capabilities") or {}


def _all_promoted_vs(capabilities: Dict[str, Any]) -> Set[str]:
    vs: Set[str] = set()
    for cluster in capabilities.values():
        for name in cluster.get("promote_value_streams", []):
            vs.add(name)
    return vs


def _all_related_vs(capabilities: Dict[str, Any]) -> Set[str]:
    vs: Set[str] = set()
    for cluster in capabilities.values():
        for name in cluster.get("related_value_streams", []):
            vs.add(name)
    return vs


# ---------------------------------------------------------------------------
# Regression checks
# ---------------------------------------------------------------------------

def check_missing_clusters(
    baseline_caps: Dict[str, Any],
    new_caps: Dict[str, Any],
) -> List[str]:
    """Return cluster keys present in baseline but missing from new map."""
    return sorted(set(baseline_caps) - set(new_caps))


def check_missing_vs_coverage(
    baseline_caps: Dict[str, Any],
    new_caps: Dict[str, Any],
) -> List[str]:
    """Return VS names that were promoted in baseline but absent from new map."""
    baseline_vs = _all_promoted_vs(baseline_caps)
    new_vs = _all_promoted_vs(new_caps)
    return sorted(baseline_vs - new_vs)


def check_cue_regressions(
    baseline_caps: Dict[str, Any],
    new_caps: Dict[str, Any],
    *,
    min_retained_fraction: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    For clusters present in both maps, check that the new cluster retains
    at least min_retained_fraction of the baseline's direct_cues.

    Returns list of regression dicts with cluster_key, dropped_cues, retained_pct.
    """
    regressions = []
    for cluster_key, baseline_cluster in baseline_caps.items():
        if cluster_key not in new_caps:
            continue
        new_cluster = new_caps[cluster_key]
        baseline_cues = set(baseline_cluster.get("direct_cues", []))
        new_cues = set(new_cluster.get("direct_cues", []))
        if not baseline_cues:
            continue
        retained = baseline_cues & new_cues
        retained_pct = len(retained) / len(baseline_cues)
        if retained_pct < min_retained_fraction:
            regressions.append({
                "cluster_key": cluster_key,
                "baseline_cue_count": len(baseline_cues),
                "new_cue_count": len(new_cues),
                "dropped_cues": sorted(baseline_cues - new_cues),
                "retained_pct": round(retained_pct * 100, 1),
            })
    return regressions


def check_weight_changes(
    baseline_caps: Dict[str, Any],
    new_caps: Dict[str, Any],
    *,
    max_weight_delta: float = 0.20,
) -> List[Dict[str, Any]]:
    """Flag clusters whose weight changed by more than max_weight_delta."""
    changes = []
    for key, baseline_cluster in baseline_caps.items():
        if key not in new_caps:
            continue
        old_w = float(baseline_cluster.get("weight", 0.9))
        new_w = float(new_caps[key].get("weight", 0.9))
        delta = abs(new_w - old_w)
        if delta > max_weight_delta:
            changes.append({
                "cluster_key": key,
                "old_weight": old_w,
                "new_weight": new_w,
                "delta": round(delta, 3),
            })
    return changes


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_regression_check(
    baseline_path: str,
    new_path: str,
    *,
    fail_on_regression: bool = True,
    min_coverage_pct: float = 90.0,
    min_cue_retained_fraction: float = 0.50,
    max_weight_delta: float = 0.20,
) -> int:
    """
    Run all regression checks and print a report.

    Returns exit code: 0 (pass), 1 (regression detected), 2 (error).
    """
    logger.info("Loading baseline: %s", baseline_path)
    baseline = _load_map(baseline_path)
    logger.info("Loading new map: %s", new_path)
    new_map = _load_map(new_path)

    baseline_caps = _get_capabilities(baseline)
    new_caps = _get_capabilities(new_map)

    logger.info(
        "Baseline: %d clusters | New: %d clusters",
        len(baseline_caps), len(new_caps),
    )

    regressions: List[str] = []
    report: Dict[str, Any] = {
        "baseline_path": baseline_path,
        "new_path": new_path,
        "baseline_cluster_count": len(baseline_caps),
        "new_cluster_count": len(new_caps),
    }

    # 1. Missing clusters
    missing_clusters = check_missing_clusters(baseline_caps, new_caps)
    report["missing_clusters"] = missing_clusters
    if missing_clusters:
        regressions.append(f"Missing {len(missing_clusters)} cluster(s): {missing_clusters}")

    # 2. Missing VS coverage
    missing_vs = check_missing_vs_coverage(baseline_caps, new_caps)
    report["missing_vs_coverage"] = missing_vs
    if missing_vs:
        regressions.append(f"Missing VS coverage for {len(missing_vs)} stream(s): {missing_vs}")

    # 3. Coverage percentage check
    baseline_vs = _all_promoted_vs(baseline_caps)
    new_vs = _all_promoted_vs(new_caps)
    if baseline_vs:
        coverage_pct = len(new_vs & baseline_vs) / len(baseline_vs) * 100
        report["vs_coverage_retained_pct"] = round(coverage_pct, 1)
        if coverage_pct < min_coverage_pct:
            regressions.append(
                f"VS coverage dropped to {coverage_pct:.1f}% (min={min_coverage_pct}%)"
            )
    else:
        report["vs_coverage_retained_pct"] = 100.0

    # 4. Cue regressions
    cue_regressions = check_cue_regressions(
        baseline_caps, new_caps,
        min_retained_fraction=min_cue_retained_fraction,
    )
    report["cue_regressions"] = cue_regressions
    if cue_regressions:
        regressions.append(
            f"{len(cue_regressions)} cluster(s) lost >50% of baseline cues"
        )

    # 5. Weight changes
    weight_changes = check_weight_changes(
        baseline_caps, new_caps,
        max_weight_delta=max_weight_delta,
    )
    report["weight_changes"] = weight_changes
    if weight_changes:
        logger.warning(
            "Weight changes detected in %d cluster(s) (informational, not a regression)",
            len(weight_changes),
        )

    # Print report
    print(json.dumps(report, indent=2))

    if regressions:
        print("\nREGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  - {r}")
        if fail_on_regression:
            return 1
    else:
        print("\nNo regressions detected.")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capability map regression checker (V6)")
    p.add_argument("--baseline", required=True, help="Path to baseline capability_map.yaml")
    p.add_argument("--new", required=True, help="Path to new capability_map.yaml to check")
    p.add_argument(
        "--fail-on-regression",
        action="store_true",
        default=True,
        help="Exit with code 1 if regressions are found (default: True)",
    )
    p.add_argument(
        "--no-fail-on-regression",
        action="store_false",
        dest="fail_on_regression",
        help="Report regressions but exit 0",
    )
    p.add_argument(
        "--min-coverage-pct",
        type=float,
        default=90.0,
        help="Minimum percent of baseline VS names that must appear in new map (default: 90.0)",
    )
    p.add_argument(
        "--min-cue-retained-fraction",
        type=float,
        default=0.50,
        help="Minimum fraction of baseline direct_cues to retain per cluster (default: 0.50)",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    code = run_regression_check(
        baseline_path=args.baseline,
        new_path=args.new,
        fail_on_regression=args.fail_on_regression,
        min_coverage_pct=args.min_coverage_pct,
        min_cue_retained_fraction=args.min_cue_retained_fraction,
    )
    sys.exit(code)
