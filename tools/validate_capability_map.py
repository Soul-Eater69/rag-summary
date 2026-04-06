"""
Capability map validator (V6 capability map pipeline — Stage 3).

Produces a comprehensive validation report for a capability_map.yaml file:

  - Coverage: which VS names from the corpus are in promote_value_streams
  - Uncovered: VS names missing from all clusters
  - Overlap: cluster pairs sharing VS names (possible redundancy)
  - Weak clusters: clusters with few or very short cues
  - Empty clusters: clusters with no promote_value_streams
  - VS-to-cluster ratio: how many clusters each VS appears in

Stage in 4-stage pipeline:
  1. tools/build_vs_corpus.py
  2. tools/build_capability_map.py
  3. tools/validate_capability_map.py   ← this file
  4. tools/capability_map_regression_check.py

Usage:
    python -m rag_summary.tools.validate_capability_map \\
        --capability-map config/capability_map.yaml \\
        [--vs-corpus data/value_stream_corpus.json] \\
        [--output-report data/capability_map_validation.json] \\
        [--fail-on-uncovered]
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_map(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_corpus(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or []


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _coverage(
    capabilities: Dict[str, Any],
    corpus: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Check how many corpus VS names are covered."""
    promoted: Set[str] = set()
    related: Set[str] = set()
    for cluster in capabilities.values():
        for vs in cluster.get("promote_value_streams", []):
            promoted.add(vs)
        for vs in cluster.get("related_value_streams", []):
            related.add(vs)

    result: Dict[str, Any] = {
        "promoted_vs_count": len(promoted),
        "related_vs_count": len(related),
        "total_unique_covered": len(promoted | related),
    }

    if corpus:
        corpus_names = {e["entity_name"] for e in corpus if e.get("entity_name")}
        uncovered_promote = corpus_names - promoted
        uncovered_all = corpus_names - (promoted | related)
        result["corpus_vs_count"] = len(corpus_names)
        result["promote_coverage_pct"] = round(len(promoted & corpus_names) / max(1, len(corpus_names)) * 100, 1)
        result["full_coverage_pct"] = round(len((promoted | related) & corpus_names) / max(1, len(corpus_names)) * 100, 1)
        result["uncovered_in_promote"] = sorted(uncovered_promote)
        result["uncovered_in_all"] = sorted(uncovered_all)

    return result


def _overlapping_clusters(capabilities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find cluster pairs that share promoted VS names (possible redundancy)."""
    vs_to_clusters: Dict[str, List[str]] = defaultdict(list)
    for cluster_key, cluster in capabilities.items():
        for vs in cluster.get("promote_value_streams", []):
            vs_to_clusters[vs].append(cluster_key)

    overlaps = []
    for vs, clusters in vs_to_clusters.items():
        if len(clusters) > 1:
            overlaps.append({"vs_name": vs, "clusters": sorted(clusters)})
    return sorted(overlaps, key=lambda x: -len(x["clusters"]))


def _weak_clusters(capabilities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flag clusters with few direct cues (< 3) or very short cues (< 4 chars)."""
    weak = []
    for key, cluster in capabilities.items():
        direct_cues = cluster.get("direct_cues", [])
        short_cues = [c for c in direct_cues if len(c) < 4]
        issues = []
        if len(direct_cues) < 3:
            issues.append(f"only {len(direct_cues)} direct cues")
        if short_cues:
            issues.append(f"{len(short_cues)} very short cues: {short_cues[:3]}")
        if not cluster.get("promote_value_streams"):
            issues.append("no promote_value_streams")
        if issues:
            weak.append({"cluster_key": key, "issues": issues})
    return weak


def _vs_cluster_ratio(capabilities: Dict[str, Any]) -> Dict[str, int]:
    """Return count of clusters each VS appears in (promote only)."""
    counts: Dict[str, int] = defaultdict(int)
    for cluster in capabilities.values():
        for vs in cluster.get("promote_value_streams", []):
            counts[vs] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(
    capability_map_path: str,
    *,
    vs_corpus_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
    fail_on_uncovered: bool = False,
) -> Dict[str, Any]:
    cap_map = _load_map(capability_map_path)
    capabilities = cap_map.get("capabilities", {})
    corpus = _load_corpus(vs_corpus_path) if vs_corpus_path else None

    logger.info("Validating %d clusters from %s", len(capabilities), capability_map_path)

    coverage = _coverage(capabilities, corpus)
    overlaps = _overlapping_clusters(capabilities)
    weak = _weak_clusters(capabilities)
    ratio = _vs_cluster_ratio(capabilities)

    report = {
        "capability_map_path": capability_map_path,
        "cluster_count": len(capabilities),
        "coverage": coverage,
        "overlapping_vs": overlaps,
        "overlapping_vs_count": len(overlaps),
        "weak_clusters": weak,
        "weak_cluster_count": len(weak),
        "vs_cluster_ratio_top10": dict(list(ratio.items())[:10]),
        "version": cap_map.get("version", "unknown"),
    }

    # Print summary
    print(json.dumps(report, indent=2))

    uncovered = coverage.get("uncovered_in_promote", [])
    if uncovered:
        print(f"\nWARNING: {len(uncovered)} VS names not in any promote_value_streams")
        for vs in uncovered[:10]:
            print(f"  - {vs}")
        if len(uncovered) > 10:
            print(f"  ... and {len(uncovered) - 10} more")

    if overlaps:
        print(f"\nINFO: {len(overlaps)} VS names appear in multiple clusters (potential redundancy)")

    if weak:
        print(f"\nWARNING: {len(weak)} weak clusters detected")

    if output_report_path:
        p = pathlib.Path(output_report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Wrote validation report → %s", output_report_path)

    if fail_on_uncovered and uncovered:
        logger.error("Failing: %d uncovered VS names", len(uncovered))
        sys.exit(1)

    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate capability map (Stage 3)")
    p.add_argument(
        "--capability-map",
        default=str(_REPO_ROOT / "config" / "capability_map.yaml"),
        help="Path to capability_map.yaml",
    )
    p.add_argument(
        "--vs-corpus",
        default=None,
        help="Path to VS corpus JSON (from build_vs_corpus.py). Required for coverage checks.",
    )
    p.add_argument(
        "--output-report",
        default=None,
        help="Optional path to write validation report JSON",
    )
    p.add_argument(
        "--fail-on-uncovered",
        action="store_true",
        help="Exit 1 if any VS names from corpus are uncovered",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    validate(
        capability_map_path=args.capability_map,
        vs_corpus_path=args.vs_corpus,
        output_report_path=args.output_report,
        fail_on_uncovered=args.fail_on_uncovered,
    )
