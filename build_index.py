"""
Build the local FAISS summary index from historical ticket chunks.

Usage:
    python -m rag_summary.build_index
    python -m rag_summary.build_index --ticket-ids IDMT-19761 IDMT-8199
    python -m rag_summary.build_index --all
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from rag_summary.ingestion.faiss_indexer import ingest_tickets_to_index, DEFAULT_INDEX_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS summary index from ticket chunks")
    parser.add_argument("--ticket-ids", nargs="+", help="Specific ticket IDs to index")
    parser.add_argument("--all", action="store_true", help="Index all tickets in ticket_chunks/")
    parser.add_argument("--ticket-chunks-dir", default="ticket_chunks", help="Ticket chunks directory")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR, help="Output FAISS index directory")
    parser.add_argument("--model", default="gpt-5-mini-idp", help="LLM model for summary generation")
    parser.add_argument("--reuse", action="store_true", default=True, help="Reuse existing summaries")
    parser.add_argument("--no-reuse", action="store_false", dest="reuse", help="Force regeneration of all summaries")
    args = parser.parse_args()

    chunks_dir = pathlib.Path(args.ticket_chunks_dir)

    if args.ticket_ids:
        ticket_ids = args.ticket_ids
    elif args.all:
        ticket_ids = sorted([
            d.name for d in chunks_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ])
    else:
        parser.print_help()
        print("\nSpecify --ticket-ids or --all")
        sys.exit(1)

    logger.info("Indexing %d tickets into %s", len(ticket_ids), args.index_dir)
    logger.info("Tickets: %s", ticket_ids)

    vectorstore = ingest_tickets_to_index(
        ticket_ids=ticket_ids,
        index_dir=args.index_dir,
        model=args.model,
        ticket_chunks_dir=args.ticket_chunks_dir,
        use_existing_summaries=args.reuse,
    )

    if vectorstore:
        logger.info("Index built successfully at %s", args.index_dir)
    else:
        logger.error("Failed to build index")
        sys.exit(1)


if __name__ == "__main__":
    main()