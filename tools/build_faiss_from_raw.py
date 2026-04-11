"""
Build FAISS index directly from fully-extracted ticket text (no chunk dependency).

This decouples FAISS indexing from the chunk generation pipeline.
You just need raw text in rag_summary_raw_text.json, prepared by
prepare_raw_text.py (which includes full attachment extraction).

The flow is:
    prepare_raw_text -> rag_summary_raw_text.json -> build_faiss_from_raw
    (Jira + attachments)    (extracted text)         (summaries + FAISS)

Usage:
    # Prepare raw text first (from prechunk cache - fastest)
    python -m rag_summary.tools.prepare_raw_text --all --from-prechunk

    # Then build FAISS from that raw text (no chunks needed)
    python -m rag_summary.tools.build_faiss_from_raw

    # Or do both in one shot:
    python -m rag_summary.tools.prepare_raw_text --all --from-prechunk && \
    python -m rag_summary.tools.build_faiss_from_raw

    # Force regenerate all summaries
    python -m rag_summary.tools.build_faiss_from_raw --no-reuse
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_RAW_TEXT_FILE = pathlib.Path("rag_summary_raw_text.json")
DEFAULT_INDEX_DIR = ROOT_DIR / "local_ticket_summary_faiss"


def load_raw_text_docs(raw_text_file: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """Load raw text JSON prepared by prepare_raw_text.py."""
    if not raw_text_file.exists():
        logger.error("Raw text file not found: %s", raw_text_file)
        logger.info("Run: python -m rag_summary.tools.prepare_raw_text --all")
        sys.exit(1)

    with open(raw_text_file, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_summary_from_raw(
    ticket_id: str,
    raw_text: str,
    title: str,
    value_stream_labels: List[str],
    model: str = "gpt-5-mini-idp",
) -> Dict[str, Any]:
    """
    Generate a semantic summary from raw ticket text.

    This is the same summary generation as the main pipeline, but
    we feed it raw text directly instead of loading from chunks.
    """
    try:
        from rag_summary.ingestion.summary_generator import generate_ticket_summary

        summary = generate_ticket_summary(
            ticket_text=raw_text,
            ticket_id=ticket_id,
            title=title,
            value_stream_labels=value_stream_labels,
            model=model,
        )
        logger.info("Generated summary for %s", ticket_id)
        return summary
    except Exception as exc:
        logger.error("Failed to generate summary for %s: %s", ticket_id, exc)
        raise

def build_faiss_from_raw_text(
    raw_text_file: pathlib.Path,
    index_dir: str,
    model: str = "gpt-5-mini-idp",
    use_existing_summaries: bool = True,
) -> bool:
    """
    Build FAISS index from raw text docs.

    1. Load raw text from JSON
    2. Generate summaries for each (or reuse existing)
    3. Build FAISS index

    Returns True on success, False otherwise.
    """
    try:
        from rag_summary.ingestion.faiss_indexer import build_summary_index
    except ImportError:
        logger.error("Could not import FAISS builder - check imports")
        return False

    # Load raw text
    raw_docs = load_raw_text_docs(raw_text_file)
    logger.info("Loaded %d raw text documents", len(raw_docs))

    # Check for existing summaries
    existing_path = os.path.join(index_dir, "summary_docs.json")
    existing_by_ticket: Dict[str, Dict[str, Any]] = {}
    if use_existing_summaries and os.path.exists(existing_path):
        try:
            with open(existing_path, encoding="utf-8") as f:
                for doc in json.load(f):
                    tid = doc.get("ticket_id", "")
                    if tid:
                        existing_by_ticket[tid] = doc
            logger.info("Loaded %d existing summaries", len(existing_by_ticket))
        except Exception as exc:
            logger.warning("Failed to load existing summaries: %s", exc)

    # Generate or reuse summaries
    summary_docs: List[Dict[str, Any]] = []
    for ticket_id, doc_data in raw_docs.items():
        if ticket_id in existing_by_ticket:
            logger.info("Reusing existing summary for %s", ticket_id)
            summary_docs.append(existing_by_ticket[ticket_id])
            continue

        try:
            summary = generate_summary_from_raw(
                ticket_id=ticket_id,
                raw_text=doc_data.get("raw_text", ""),
                title=doc_data.get("title", ticket_id),
                value_stream_labels=doc_data.get("value_stream_labels", []),
                model=model,
            )
            summary_docs.append(summary)
        except Exception as exc:
            logger.error("Failed for %s: %s", ticket_id, exc)
            continue

    if not summary_docs:
        logger.error("No summaries generated, cannot build index")
        return False

    # Build index
    try:
        build_summary_index(summary_docs, index_dir=index_dir)
        logger.info("Index built successfully at %s", index_dir)
        return True
    except Exception as exc:
        logger.error("FAISS index build failed: %s", exc)
        return False

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from raw ticket text (no chunks needed)"
    )
    parser.add_argument(
        "--raw-text-file",
        type=pathlib.Path,
        default=DEFAULT_RAW_TEXT_FILE,
        help=f"Path to raw text JSON (default: {DEFAULT_RAW_TEXT_FILE})",
    )
    parser.add_argument(
        "--index-dir",
        type=pathlib.Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Output FAISS index directory (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini-idp",
        help="LLM model for summary generation (default: gpt-5-mini-idp)",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Force regeneration of all summaries (don't reuse existing)",
    )

    args = parser.parse_args()

    logger.info("Building FAISS index from raw text...")
    success = build_faiss_from_raw_text(
        raw_text_file=args.raw_text_file,
        index_dir=str(args.index_dir),
        model=args.model,
        use_existing_summaries=not args.no_reuse,
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
