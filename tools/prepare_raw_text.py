"""
Prepare fully-extracted ticket text independently from chunk generation.

This decouples Jira extraction from FAISS indexing. It runs the same full
ingestion pipeline (description + comments + attachment extraction/OCR)
but saves the **pre-chunk extracted text** - not raw Jira JSON and not
the chunked output.

The output is a single JSON file that the FAISS builder can consume
without touching Jira or the chunk pipeline again.

Usage:
    # Full extraction for specific tickets (attachments, OCR, everything)
    python -m rag_summary.tools.prepare_raw_text --tickets IDMT-19761 IDMT-8199

    # Full extraction for default ticket list
    python -m rag_summary.tools.prepare_raw_text --all

    # Skip tickets that already have an entry in the output file
    python -m rag_summary.tools.prepare_raw_text --all --incremental

    # Use existing ingest results from jira_prechunk_output/ (no Jira call)
    python -m rag_summary.tools.prepare_raw_text --all --from-prechunk

    # Dry-run (print what would happen)
    python -m rag_summary.tools.prepare_raw_text --tickets IDMT-19761 --dry-run

Output:
    rag_summary_raw_text.json with structure:
    {
        "IDMT-19761": {
            "ticket_id": "IDMT-19761",
            "title": "...",
            "raw_text": "full extracted text (description + attachments + comments)",
            "description": "...",
            "primary_attachment_text": "...",
            "comments_cleaned": ["..."],
            "value_stream_labels": [...],
            "extraction_source": "jira_live" | "prechunk_cache"
        },
        ...
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional

from ingestion.value_stream_mapping_service import canonicalize_value_stream_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_OUTPUT = ROOT_DIR / "rag_summary_raw_text.json"
DEFAULT_PRECHUNK_DIR = ROOT_DIR / "jira_prechunk_output"


def _normalize_vs_labels(names: List[str] | None) -> List[str]:
    """
    Prefer Azure-canonicalized value stream names, but preserve source names
    when canonicalization returns empty (e.g., project/theme labels outside
    the taxonomy registry).
    """
    source_names = [str(name).strip() for name in (names or []) if str(name).strip()]
    canonical = canonicalize_value_stream_names(source_names)
    if canonical:
        return canonical

    deduped: List[str] = []
    seen = set()
    for name in source_names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped

# Default ticket list (mirrors batch_ingest_job)
DEFAULT_TICKETS = [
    "IDMT-1320",
    "IDMT-4125",
    "IDMT-4124",
    "IDMT-14403",
    "IDMT-12760",
    "IDMT-12362",
    "IDMT-8280",
    "IDMT-8199",
    "IDMT-9431",
    "IDMT-8281",
    "IDMT-12129",
    "IDMT-23229",
    "IDMT-19761",
    "IDMT-31170",
    "IDMT-30892",
]

# ----------------------------------------------------------------------
# Extract from existing prechunk ingest results (no Jira call needed)
# ----------------------------------------------------------------------

def extract_from_prechunk(ticket_id: str, prechunk_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    """
    Read the fully-extracted ingest result (00_ingest_result.json) from
    jira_prechunk_output/ and pull out the pre-chunk text fields.

    This is the fastest path - no Jira API call, no attachment download.
    The ingest pipeline already ran; we just read the result.
    """
    ticket_dir = prechunk_dir / ticket_id
    ingest_file = ticket_dir / "00_ingest_result.json"

    if not ingest_file.exists():
        # Fall back to ticket_chunks if prechunk not available
        for alt_dir in [ROOT_DIR / "ticket_chunks" / ticket_id]:
            for candidate in ["07_chunks.json", "01_ticket_data.json"]:
                alt_file = alt_dir / candidate
                if alt_file.exists():
                    return _extract_from_chunk_file(ticket_id, alt_file)
        return None

    try:
        with open(ingest_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        observed = result.get("observed", {})
        raw = result.get("raw", {})
        supervision = result.get("supervision", {})

        # The retrieval_text is the best single-text representation
        # It includes: summary + description + primary attachment + supporting docs
        retrieval_text = observed.get("retrieval_text", "")

        # Also grab individual components for flexibility
        description = observed.get("description_cleaned", "")
        primary_att_text = observed.get("primary_attachment_text", "")
        comments_cleaned = observed.get("comments_cleaned", [])
        title = (observed.get("metadata") or {}).get("summary", ticket_id)

        # VS labels from supervision layer (dedupe + canonicalization with fallback)
        vs_labels = _normalize_vs_labels(supervision.get("linked_value_stream_names", []))

        # Build comprehensive raw_text from all extracted sources
        raw_text = _assemble_raw_text(
            retrieval_text=retrieval_text,
            description=description,
            primary_attachment_text=primary_att_text,
            comments=comments_cleaned,
        )

        return {
            "ticket_id": ticket_id,
            "title": title,
            "raw_text": raw_text,
            "description": description,
            "primary_attachment_text": primary_att_text[:8000] if primary_att_text else "",
            "comments_cleaned": comments_cleaned[:5],
            "value_stream_labels": vs_labels,
            "extraction_source": "prechunk_cache",
            "char_count": len(raw_text),
        }

    except Exception as exc:
        logger.error("Failed to read prechunk for %s: %s", ticket_id, exc)
        return None

def _extract_from_chunk_file(ticket_id: str, chunk_file: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Fallback: extract text from 07_chunks.json or similar."""
    try:
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        texts = []
        for chunk in chunks:
            text = chunk.get("text") or chunk.get("content") or chunk.get("retrieval_text") or ""
            if text and len(text) > 20:
                texts.append(text)

        raw_text = "\n\n".join(texts)
        vs_labels = _normalize_vs_labels(data.get("mapped_value_stream_names", []))

        return {
            "ticket_id": ticket_id,
            "title": ticket_id,
            "raw_text": raw_text,
            "description": "",
            "primary_attachment_text": "",
            "comments_cleaned": [],
            "value_stream_labels": vs_labels,
            "extraction_source": "chunk_fallback",
            "char_count": len(raw_text),
        }
    except Exception as exc:
        logger.error("Failed to read chunk file for %s: %s", ticket_id, exc)
        return None

# ----------------------------------------------------------------------
# Live Jira extraction (full pipeline, with attachments)
# ----------------------------------------------------------------------

async def extract_from_jira_live(ticket_id: str) -> Optional[Dict[str, Any]]:
    """
    Run the full ingestion pipeline against live Jira to get fully-extracted
    text including attachment OCR/conversion.

    This is slower but gives the freshest data.
    """
    try:
        from jira_ingestion import JiraValueStreamClient
        from jira_ingestion.runtime.runtime_factory import build_ingestion_config
        from src.config import JIRA_BASE_URL, JIRA_TOKEN
        from ingestion.pipeline import ingest_ticket_payload

        cfg = build_ingestion_config(
            llm_model="gpt-4o-mini-idp",
            skip_llm_summary=True,        # We don't need LLM summary for raw text
            skip_llm_keywords=True,       # Skip keywords too
            skip_llm_derived=True,        # Skip derived artifacts
        )

        async with JiraValueStreamClient(
            base_url=JIRA_BASE_URL,
            token=JIRA_TOKEN,
            verify_ssl=False,
        ) as jira_client:
            ticket_data = await jira_client.get_ticket_data(ticket_id, config=cfg)
            result = await ingest_ticket_payload(
                ticket_data=ticket_data,
                jira_client=jira_client,
                llm_client=None,
                embedding_client=None,
                config=cfg,
            )

        observed = result.get("observed", {})
        supervision = result.get("supervision", {})

        retrieval_text = observed.get("retrieval_text", "")
        description = observed.get("description_cleaned", "")
        primary_att_text = observed.get("primary_attachment_text", "")
        comments_cleaned = observed.get("comments_cleaned", [])
        title = (observed.get("metadata") or {}).get("summary", ticket_id)
        
        vs_labels = _normalize_vs_labels(supervision.get("linked_value_stream_names", []))
        if not vs_labels:
            live_vs_names = [
                (entry or {}).get("name", "")
                for entry in (ticket_data.get("value_streams") or [])
                if (entry or {}).get("name")
            ]
            vs_labels = _normalize_vs_labels(live_vs_names)

        raw_text = _assemble_raw_text(
            retrieval_text=retrieval_text,
            description=description,
            primary_attachment_text=primary_att_text,
            comments=comments_cleaned,
        )

        return {
            "ticket_id": ticket_id,
            "title": title,
            "raw_text": raw_text,
            "description": description,
            "primary_attachment_text": primary_att_text[:8000] if primary_att_text else "",
            "comments_cleaned": comments_cleaned[:5],
            "value_stream_labels": vs_labels,
            "extraction_source": "jira_live",
            "char_count": len(raw_text),
        }

    except Exception as exc:
        logger.error("Failed live extraction for %s: %s", ticket_id, exc)
        return None

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _assemble_raw_text(
    retrieval_text: str,
    description: str,
    primary_attachment_text: str,
    comments: List[str],
) -> str:
    """
    Build the best single text block from all extracted sources.

    Priority: use retrieval_text if available (it's already the best
    composite). Otherwise fall back to concatenating components.
    """
    if retrieval_text and len(retrieval_text) > 100:
        return retrieval_text

    parts = []
    if description:
        parts.append(description)
    if primary_attachment_text:
        parts.append(primary_attachment_text)
    for comment in (comments or [])[:3]:
        if comment and len(comment) > 30:
            parts.append(comment)
    return "\n\n".join(parts)

# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def prepare_all_tickets(
    ticket_ids: List[str],
    output_file: pathlib.Path,
    *,
    from_prechunk: bool = False,
    prechunk_dir: pathlib.Path = DEFAULT_PRECHUNK_DIR,
    incremental: bool = False,
    dry_run: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch/extract raw text for all tickets and optionally save to file."""
    # Load existing if incremental
    existing: Dict[str, Dict[str, Any]] = {}
    if incremental and output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            logger.info("Loaded %d existing entries (incremental mode)", len(existing))
        except Exception:
            pass

    all_docs: Dict[str, Dict[str, Any]] = dict(existing)

    for ticket_id in ticket_ids:
        if incremental and ticket_id in existing:
            logger.info("Skipping %s (already in output)", ticket_id)
            continue

        logger.info("Extracting %s ...", ticket_id)

        if from_prechunk:
            doc = extract_from_prechunk(ticket_id, prechunk_dir)
        else:
            doc = asyncio.run(extract_from_jira_live(ticket_id))

        if doc:
            all_docs[ticket_id] = doc
            logger.info("  OK: %d chars from %s", doc.get("char_count", 0), doc.get("extraction_source", "?"))
        else:
            logger.warning("  FAILED: no text extracted for %s", ticket_id)

    if not dry_run and all_docs:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d raw text entries to %s", len(all_docs), output_file)
    elif dry_run:
        logger.info("[DRY-RUN] Would save %d tickets to %s", len(all_docs), output_file)
        for tid, doc in all_docs.items():
            logger.info("  %s: %d chars, source=%s", tid, doc.get("char_count", 0), doc.get("extraction_source", "?"))

    return all_docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract fully-processed ticket text (with attachments) for FAISS indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Extract from existing prechunk cache (fastest, no Jira call)
  python -m rag_summary.tools.prepare_raw_text --all --from-prechunk

  # Extract fresh from Jira (full attachment processing)
  python -m rag_summary.tools.prepare_raw_text --tickets IDMT-19761 IDMT-8199

  # Only extract tickets not already in the output file
  python -m rag_summary.tools.prepare_raw_text --all --from-prechunk --incremental
"""
    )
    parser.add_argument("--tickets", nargs="+", metavar="TICKET_ID", help="Specific ticket IDs")
    parser.add_argument("--all", action="store_true", help="Use default ticket list")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT, help=f"Output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--from-prechunk", action="store_true", help="Read from jira_prechunk_output/ instead of live Jira")
    parser.add_argument("--prechunk-dir", type=pathlib.Path, default=DEFAULT_PRECHUNK_DIR, help="Prechunk output directory")
    parser.add_argument("--incremental", action="store_true", help="Skip tickets already in output file")
    parser.add_argument("--dry-run", action="store_true", help="Print without saving")

    args = parser.parse_args()

    if args.tickets:
        ticket_ids = [t.upper() for t in args.tickets]
    elif args.all:
        ticket_ids = DEFAULT_TICKETS
    else:
        parser.print_help()
        print("\nSpecify --tickets or --all")
        sys.exit(1)

    logger.info("Preparing %d tickets", len(ticket_ids))
    prepare_all_tickets(
        ticket_ids,
        args.output,
        from_prechunk=args.from_prechunk,
        prechunk_dir=args.prechunk_dir,
        incremental=args.incremental,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
