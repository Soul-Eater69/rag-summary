"""
FAISS indexer: build, load, search, and batch-populate the local summary FAISS index.

Storage layout:
  <index_dir>/
    index.faiss
    index.pkl
    summary_docs.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.clients.embedding import EmbeddingClient
from .summary_generator import (
    build_retrieval_text,
    generate_historical_ticket_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = str(
    pathlib.Path(__file__).resolve().parent.parent.parent / "local_ticket_summary_faiss"
)

# -----------------------------------------------------------------------------
# Document conversion
# -----------------------------------------------------------------------------

def _make_document(summary_doc: Dict[str, Any]) -> Document:
    """Convert a summary dict into a LangChain Document for FAISS."""
    retrieval_text = build_retrieval_text(summary_doc)
    metadata = {
        "doc_id": summary_doc.get("doc_id", ""),
        "ticket_id": summary_doc.get("ticket_id", ""),
        "title": summary_doc.get("title", ""),
        "short_summary": summary_doc.get("short_summary", ""),
        "business_goal": summary_doc.get("business_goal", ""),
        "actors": summary_doc.get("actors", []),
        "direct_functions": summary_doc.get("direct_functions", []),
        "implied_functions": summary_doc.get("implied_functions", []),
        "change_types": summary_doc.get("change_types", []),
        "domain_tags": summary_doc.get("domain_tags", []),
        "evidence_sentences": summary_doc.get("evidence_sentences", []),
        "value_stream_labels": summary_doc.get("value_stream_labels", []),
        "doc_type": "ticket_summary",
    }
    return Document(page_content=retrieval_text, metadata=metadata)


# -----------------------------------------------------------------------------
# Index operations
# -----------------------------------------------------------------------------

def build_summary_index(
    summary_docs: List[Dict[str, Any]],
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Any:
    """Build a FAISS index from a list of summary dicts and persist to disk."""
    from langchain_community.vectorstores import FAISS

    os.makedirs(index_dir, exist_ok=True)
    documents = [_make_document(doc) for doc in summary_docs]
    embeddings = EmbeddingClient()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_dir)

    # Persist raw summary docs for debugging/inspection
    with open(os.path.join(index_dir, "summary_docs.json"), "w", encoding="utf-8") as f:
        json.dump(summary_docs, f, ensure_ascii=False, indent=2)

    # Persist index manifest to detect stale summaries (changed prompt/schema/model)
    manifest = {
        "summary_prompt_version": "1",
        "retrieval_text_packing_version": "1",
        "embedding_model": embeddings.model if hasattr(embeddings, "model") else "unknown",
        "ticket_count": len(summary_docs),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(index_dir, "index_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Built FAISS summary index with %d documents at %s", len(documents), index_dir)
    return vectorstore


def load_summary_index(index_dir: str = DEFAULT_INDEX_DIR) -> Any:
    """Load a previously persisted FAISS summary index."""
    from langchain_community.vectorstores import FAISS

    embeddings = EmbeddingClient()
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def search_summary_index(
    query_text: str,
    index_dir: str = DEFAULT_INDEX_DIR,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search the summary FAISS index and return ranked analog tickets.

    Score semantics: FAISS returns L2 distance (lower = more similar). We
    normalise to a similarity score via  ``1 / (1 + distance)`` so that
    **higher score always means more relevant** throughout the rest of the
    pipeline.  Scores are in (0, 1].
    """
    vectorstore = load_summary_index(index_dir)
    results = vectorstore.similarity_search_with_score(query_text, k=top_k)

    out: List[Dict[str, Any]] = []
    for rank, (doc, raw_distance) in enumerate(results, start=1):
        # Normalise: higher is always better (L2 distance → similarity)
        similarity = round(1.0 / (1.0 + float(raw_distance)), 4)
        out.append({
            "rank": rank,
            "score": similarity,
            "ticket_id": doc.metadata.get("ticket_id", ""),
            "title": doc.metadata.get("title", ""),
            "short_summary": doc.metadata.get("short_summary", ""),
            "business_goal": doc.metadata.get("business_goal", ""),
            "actors": doc.metadata.get("actors", []),
            "direct_functions": doc.metadata.get("direct_functions", []),
            "implied_functions": doc.metadata.get("implied_functions", []),
            "change_types": doc.metadata.get("change_types", []),
            "domain_tags": doc.metadata.get("domain_tags", []),
            "value_stream_labels": doc.metadata.get("value_stream_labels", []),
            "retrieval_text": doc.page_content,
        })

    return out


# -----------------------------------------------------------------------------
# Batch ingestion: generate summaries for historical tickets -> build index
# -----------------------------------------------------------------------------

def ingest_tickets_to_index(
    ticket_ids: List[str],
    *,
    index_dir: str = DEFAULT_INDEX_DIR,
    model: str = "gpt-5-mini-idp",
    ticket_chunks_dir: str = "ticket_chunks",
    use_existing_summaries: bool = True,
) -> Any:
    """
    For each ticket:
    1. Load retrieval text from ticket chunks (or ingest from Jira if needed)
    2. Load known value-stream labels from 08_valuestream_map.json
    3. Generate semantic summary via LLM
    4. Build FAISS index from all summaries

    Returns the built FAISS vectorstore.
    """
    chunks_path = pathlib.Path(ticket_chunks_dir)
    summary_docs: List[Dict[str, Any]] = []

    # Check for existing summary docs
    existing_path = os.path.join(index_dir, "summary_docs.json")
    existing_by_ticket: Dict[str, Dict[str, Any]] = {}
    if use_existing_summaries and os.path.exists(existing_path):
        try:
            with open(existing_path, encoding="utf-8") as f:
                for doc in json.load(f):
                    tid = doc.get("ticket_id", "")
                    if tid:
                        existing_by_ticket[tid] = doc
            logger.info("Loaded %d existing summaries from %s", len(existing_by_ticket), existing_path)
        except Exception:
            pass

    for ticket_id in ticket_ids:
        # Reuse existing summary if available
        if ticket_id in existing_by_ticket:
            logger.info("Reusing existing summary for %s", ticket_id)
            summary_docs.append(existing_by_ticket[ticket_id])
            continue

        ticket_dir = chunks_path / ticket_id

        # Load retrieval text from all_chunks or concatenated chunk files
        retrieval_text = _load_ticket_retrieval_text(ticket_dir)
        if not retrieval_text:
            logger.warning("No retrieval text found for %s, skipping", ticket_id)
            continue

        # Load known value-stream labels
        vs_labels = _load_ticket_vs_labels(ticket_dir)

        # Load title
        title = _load_ticket_title(ticket_dir, ticket_id)

        # Generate summary
        try:
            summary = generate_historical_ticket_summary(
                ticket_text=retrieval_text,
                ticket_id=ticket_id,
                title=title,
                value_stream_labels=vs_labels,
                model=model,
            )
            summary_docs.append(summary)
            logger.info("Generated summary for %s", ticket_id)
        except Exception as exc:
            logger.error("Failed to generate summary for %s: %s", ticket_id, exc)

    if not summary_docs:
        logger.warning("No summaries generated, cannot build index")
        return None

    return build_summary_index(summary_docs, index_dir=index_dir)


def _load_ticket_retrieval_text(ticket_dir: pathlib.Path) -> str:
    """Load concatenated text from ticket chunk files."""
    # Try the chunk files in order - pick the richest available source
    for candidate in [
        "03_retrieval_views.json",
        "02_attachment_text.json",
        "01_ticket_data.json",
    ]:
        path = ticket_dir / candidate
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Concatenate all text-like values
                    parts = []
                    for key, val in data.items():
                        if isinstance(val, str) and len(val) > 20:
                            parts.append(val)
                        elif isinstance(val, list):
                            for item in val:
                                if isinstance(item, dict):
                                    text = item.get("text") or item.get("content") or item.get("retrieval_text") or ""
                                    if text:
                                        parts.append(str(text))
                                elif isinstance(item, str) and len(item) > 20:
                                    parts.append(item)
                    if parts:
                        return "\n\n".join(parts)
                elif isinstance(data, str):
                    return data
            except Exception:
                continue
    return ""


def _load_ticket_vs_labels(ticket_dir: pathlib.Path) -> List[str]:
    """Load value-stream labels from the ticket's 08_valuestream_map.json."""
    vs_map_path = ticket_dir / "08_valuestream_map.json"
    if not vs_map_path.exists():
        return []
    try:
        with open(vs_map_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("valueStreamNames", []) or []
    except Exception:
        return []


def _load_ticket_title(ticket_dir: pathlib.Path, fallback: str) -> str:
    """Load ticket title from 08_valuestream_map.json or 01_ticket_data.json."""
    for candidate in ["08_valuestream_map.json", "01_ticket_data.json"]:
        path = ticket_dir / candidate
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                title = data.get("title") or data.get("summary") or ""
                if title:
                    # Strip "TICKETID: " prefix
                    return title.split(": ", 1)[1] if ": " in title else title
            except Exception:
                continue
    return fallback