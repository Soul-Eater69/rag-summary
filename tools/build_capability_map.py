"""
Evidence-driven capability map builder (V6).

Replaces the template-only bootstrap_capability_map.py approach with one
driven by actual historical ticket evidence. Rather than relying on hand-crafted
cue lists, this tool mines real ticket summaries to discover which cue phrases
co-occur with each value stream, producing a capability map grounded in evidence.

Flow:
  1. Load ticket summaries (same format used by build_theme_index.py)
  2. For each VS name, collect all tickets that include it in value_stream_labels
  3. Extract top cue phrases from ticket retrieval_text via TF-IDF or word frequency
  4. Identify co-occurring canonical_functions and capability_tags
  5. Write/update capability_map.yaml with evidence-grounded cues

The output is compatible with the existing capability_map.yaml schema so it
can be used as a drop-in replacement for the bootstrap-generated map.

Usage:
    python -m summary_rag.tools.build_capability_map \\
        --summary-dir summaries/ \\
        --output-path config/capability_map.yaml \\
        [--min-ticket-count 3] \\
        [--top-cues 15] \\
        [--cutoff-date 2024-01-01] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Ticket summary loading (shared with build_theme_index.py)
# ---------------------------------------------------------------------------

def _load_ticket_summaries(summary_dir: str, cutoff_date: Optional[str]) -> List[Dict[str, Any]]:
    """Load JSON ticket summaries with optional cutoff date filter."""
    summaries = []
    if not os.path.isdir(summary_dir):
        logger.error("summary_dir does not exist: %s", summary_dir)
        sys.exit(1)
    for fname in sorted(os.listdir(summary_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(summary_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as exc:
            logger.warning("Skipping %s: %s", fname, exc)
            continue
        if cutoff_date:
            ticket_date = doc.get("ingested_at") or doc.get("created_at") or ""
            if ticket_date and ticket_date > cutoff_date:
                continue
        summaries.append(doc)
    logger.info("Loaded %d ticket summaries", len(summaries))
    return summaries


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def _extract_cue_phrases(
    tickets: List[Dict[str, Any]],
    *,
    top_n: int = 15,
    min_word_len: int = 4,
    stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Extract top cue phrases (unigrams and bigrams) from ticket retrieval_text
    using term frequency, weighted by ticket similarity score if present.
    """
    default_stopwords = {
        "this", "that", "with", "from", "will", "have", "been",
        "they", "their", "would", "could", "should", "also", "into",
        "using", "used", "based", "which", "when", "then", "than",
    }
    stops = stopwords or default_stopwords

    word_scores: Counter = Counter()
    bigram_scores: Counter = Counter()

    for ticket in tickets:
        weight = max(0.1, float(ticket.get("score", 1.0)))
        text = ticket.get("retrieval_text", "").lower()
        words = [w.strip(".,;:!?()'\"") for w in text.split()]
        words = [w for w in words if len(w) >= min_word_len and w not in stops]
        for w in words:
            word_scores[w] += weight
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigram_scores[bigram] += weight * 0.8  # bigrams weighted slightly less

    # Merge and deduplicate
    combined: Counter = Counter()
    combined.update(word_scores)
    combined.update(bigram_scores)

    return [phrase for phrase, _ in combined.most_common(top_n)]


def _extract_canonical_functions(
    tickets: List[Dict[str, Any]],
    *,
    top_n: int = 10,
) -> List[str]:
    """Extract most common canonical functions across a set of tickets."""
    func_counter: Counter = Counter()
    for ticket in tickets:
        for fn in ticket.get("canonical_functions", []) or ticket.get("direct_functions_canonical", []):
            if fn:
                func_counter[fn.strip()] += 1
    return [fn for fn, _ in func_counter.most_common(top_n)]


def _extract_capability_tags(
    tickets: List[Dict[str, Any]],
    *,
    top_n: int = 10,
) -> List[str]:
    """Extract most common capability_tags across a set of tickets."""
    tag_counter: Counter = Counter()
    for ticket in tickets:
        for tag in ticket.get("capability_tags", []):
            if tag:
                tag_counter[tag.strip()] += 1
    return [tag for tag, _ in tag_counter.most_common(top_n)]


# ---------------------------------------------------------------------------
# VS-to-cluster mapping
# ---------------------------------------------------------------------------

def _build_vs_ticket_map(
    tickets: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group tickets by the VS names they contain."""
    vs_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ticket in tickets:
        for vs in ticket.get("value_stream_labels", []):
            if vs:
                vs_map[vs].append(ticket)
    return dict(vs_map)


# ---------------------------------------------------------------------------
# Cluster name derivation (simple slug)
# ---------------------------------------------------------------------------

def _vs_to_cluster_name(vs_name: str) -> str:
    """Convert VS name to a snake_case cluster key."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", vs_name.lower().strip()).strip("_")
    return slug[:60]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_capability_map(
    summary_dir: str,
    output_path: str,
    *,
    min_ticket_count: int = 3,
    top_cues: int = 15,
    cutoff_date: Optional[str] = None,
    dry_run: bool = False,
    existing_map_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build evidence-driven capability_map.yaml.

    For each VS that has >= min_ticket_count historical tickets:
      - Derives a cluster key from the VS name
      - Extracts direct cues (word-freq based) as direct_cues
      - Extracts canonical functions
      - Sets promote_value_streams = [vs_name]

    If existing_map_path is provided, preserves any clusters not covered by
    evidence (keeps hand-crafted entries for rare VS names).

    Returns the capability map dict.
    """
    tickets = _load_ticket_summaries(summary_dir, cutoff_date)
    if not tickets:
        logger.error("No tickets loaded — cannot build capability map")
        sys.exit(1)

    vs_ticket_map = _build_vs_ticket_map(tickets)
    logger.info("Found %d unique VS names in ticket data", len(vs_ticket_map))

    # Load existing map to preserve hand-crafted clusters
    existing_capabilities: Dict[str, Any] = {}
    if existing_map_path and os.path.exists(existing_map_path):
        try:
            with open(existing_map_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
            existing_capabilities = existing.get("capabilities", {})
            logger.info("Loaded %d existing clusters from %s", len(existing_capabilities), existing_map_path)
        except Exception as exc:
            logger.warning("Could not load existing map: %s", exc)

    capabilities: Dict[str, Any] = {}

    # Start from existing to preserve any manual overrides
    capabilities.update(existing_capabilities)

    evidence_count = 0
    for vs_name, vs_tickets in sorted(vs_ticket_map.items()):
        if len(vs_tickets) < min_ticket_count:
            logger.debug("Skipping %s: only %d tickets (min=%d)", vs_name, len(vs_tickets), min_ticket_count)
            continue

        cluster_key = _vs_to_cluster_name(vs_name)
        direct_cues = _extract_cue_phrases(vs_tickets, top_n=top_cues)
        canonical_functions = _extract_canonical_functions(vs_tickets)
        capability_tags = _extract_capability_tags(vs_tickets)

        # Split cues: top half as direct, rest as indirect
        mid = max(1, len(direct_cues) // 2)
        direct = direct_cues[:mid]
        indirect = direct_cues[mid:]

        # Merge with existing cluster if present (preserving hand-crafted cues)
        existing_cluster = existing_capabilities.get(cluster_key, {})
        existing_direct = set(existing_cluster.get("direct_cues", []))
        existing_indirect = set(existing_cluster.get("indirect_cues", []))
        existing_funcs = set(existing_cluster.get("canonical_functions", []))

        merged_direct = list(existing_direct | set(direct))[:top_cues]
        merged_indirect = list(existing_indirect | set(indirect))[:top_cues]
        merged_funcs = list(existing_funcs | set(canonical_functions))[:10]

        capabilities[cluster_key] = {
            "description": existing_cluster.get(
                "description",
                f"Evidence-driven cluster for {vs_name} ({len(vs_tickets)} tickets).",
            ),
            "direct_cues": merged_direct,
            "indirect_cues": merged_indirect,
            "canonical_functions": merged_funcs,
            "capability_tags": capability_tags[:10],
            "promote_value_streams": existing_cluster.get("promote_value_streams", [vs_name]),
            "related_value_streams": existing_cluster.get("related_value_streams", []),
            "weight": existing_cluster.get("weight", 0.9),
            "evidence_ticket_count": len(vs_tickets),
        }
        evidence_count += 1

    logger.info("Built %d evidence-backed clusters (preserved %d existing)", evidence_count, len(existing_capabilities))

    capability_map = {
        "version": 2,
        "built_from_evidence": True,
        "source_ticket_count": len(tickets),
        "capabilities": capabilities,
    }

    if dry_run:
        print(yaml.dump(capability_map, default_flow_style=False, sort_keys=False))
        return capability_map

    output_path_obj = pathlib.Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        yaml.dump(capability_map, f, default_flow_style=False, sort_keys=False)
    logger.info("Wrote capability map to %s (%d clusters)", output_path, len(capabilities))
    return capability_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build evidence-driven capability map (V6)")
    p.add_argument("--summary-dir", required=True, help="Directory of ticket JSON summaries")
    p.add_argument(
        "--output-path",
        default=str(_REPO_ROOT / "config" / "capability_map.yaml"),
        help="Output path for capability_map.yaml",
    )
    p.add_argument("--min-ticket-count", type=int, default=3, help="Min tickets per VS to create a cluster (default: 3)")
    p.add_argument("--top-cues", type=int, default=15, help="Number of cue phrases to extract per VS (default: 15)")
    p.add_argument("--cutoff-date", default=None, help="ISO-8601 date; exclude tickets after this date")
    p.add_argument("--existing-map-path", default=None, help="Existing capability_map.yaml to merge with")
    p.add_argument("--dry-run", action="store_true", help="Print result without writing")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    build_capability_map(
        summary_dir=args.summary_dir,
        output_path=args.output_path,
        min_ticket_count=args.min_ticket_count,
        top_cues=args.top_cues,
        cutoff_date=args.cutoff_date,
        dry_run=args.dry_run,
        existing_map_path=args.existing_map_path,
    )
