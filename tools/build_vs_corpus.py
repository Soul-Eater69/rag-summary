"""
VS corpus builder (V6 capability map pipeline — Stage 1).

Fetches canonical value stream entries from the KG retrieval service and
writes a normalized corpus JSON for use by subsequent capability map stages.

This is the first stage of the 4-stage capability map pipeline:
  1. build_vs_corpus.py          ← this file
  2. tools/build_capability_map.py  (evidence-driven draft)
  3. tools/validate_capability_map.py (coverage + quality checks)
  4. tools/capability_map_regression_check.py (regression guard)

Output (--output-path):
  JSON array of VS corpus entries:
    {"entity_id", "entity_name", "description", "value_proposition",
     "aliases", "category"}

Usage:
    python -m rag_summary.tools.build_vs_corpus \\
        --output-path data/value_stream_corpus.json \\
        [--top-k 200] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "value_stream_corpus.json"


def fetch_from_kg(*, top_k: int = 200) -> List[Dict[str, Any]]:
    """
    Fetch canonical VS entries from the KG retrieval service.

    Requires src.pipelines.value_stream.retrieval_pipeline to be available.
    Returns a normalized corpus list.
    """
    try:
        from src.pipelines.value_stream.retrieval_pipeline import (  # type: ignore[import]
            retrieve_kg_candidates as _retrieve,
        )
        # Broad empty-string query to retrieve the full VS catalog
        raw = _retrieve("", top_k=top_k, allowed_names=None)
    except ImportError:
        logger.error(
            "KG retrieval pipeline not available. "
            "Run inside the internal codebase environment or pass --from-file."
        )
        sys.exit(2)
    except Exception as exc:
        logger.error("KG retrieval failed: %s", exc)
        sys.exit(2)

    corpus: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for entry in raw:
        name = (entry.get("entity_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        corpus.append({
            "entity_id": entry.get("entity_id", ""),
            "entity_name": name,
            "description": entry.get("description", ""),
            "value_proposition": entry.get("value_proposition", ""),
            "aliases": entry.get("aliases", []),
            "category": entry.get("category", ""),
        })

    logger.info("Fetched %d unique VS entries from KG", len(corpus))
    return corpus


def load_from_file(path: str) -> List[Dict[str, Any]]:
    """Load an existing VS corpus JSON file (for offline / re-validation runs)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")
    logger.info("Loaded %d VS entries from %s", len(data), path)
    return data


def write_corpus(corpus: List[Dict[str, Any]], output_path: str) -> None:
    p = pathlib.Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    logger.info("Wrote VS corpus → %s (%d entries)", output_path, len(corpus))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build VS corpus from KG (Stage 1)")
    p.add_argument("--output-path", default=str(_DEFAULT_OUTPUT), help="Output JSON path")
    p.add_argument("--from-file", default=None, help="Load from existing corpus file instead of KG")
    p.add_argument("--top-k", type=int, default=200, help="Max VS entries to fetch from KG")
    p.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    if args.from_file:
        corpus = load_from_file(args.from_file)
    else:
        corpus = fetch_from_kg(top_k=args.top_k)

    if args.dry_run:
        print(f"Would write {len(corpus)} VS entries to {args.output_path}")
        for entry in corpus[:5]:
            print(f"  - {entry['entity_name']} ({entry.get('category', 'unknown')})")
        if len(corpus) > 5:
            print(f"  ... and {len(corpus) - 5} more")
    else:
        write_corpus(corpus, args.output_path)
