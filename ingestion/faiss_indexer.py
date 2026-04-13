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
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .adapters import EmbeddingService, get_default_embedding
from .summary_generator import (
    build_retrieval_text,
    generate_ticket_summary,
)
from .summary_loader import (
    load_ticket_retrieval_text,
    load_ticket_vs_labels,
    load_ticket_title,
)

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = str(
    pathlib.Path(__file__).resolve().parent.parent.parent / "local_ticket_summary_faiss"
)


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive threading bridge
            error["value"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value")


async def _load_ticket_inputs_from_jira(ticket_ids: List[str], model: str) -> Dict[str, Dict[str, Any]]:
    from ingestion.ticket_ingestion_service import IngestionDeps, ingest_one_ticket
    from jira.value_stream_client import JiraValueStreamClient
    from jira_ingestion.runtime.runtime_factory import build_ingestion_config

    jira_base_url = os.environ.get("JIRA_BASE_URL", "").rstrip("/")
    jira_token = os.environ.get("JIRA_TOKEN", "")
    if not jira_base_url or not jira_token:
        raise RuntimeError("Missing JIRA_BASE_URL or JIRA_TOKEN")

    cfg = build_ingestion_config(
        llm_model=model,
        skip_llm_summary=True,
        skip_llm_keywords=False,
        skip_llm_derived=True,
    )

    payload_by_ticket: Dict[str, Dict[str, Any]] = {}
    async with JiraValueStreamClient(base_url=jira_base_url, token=jira_token, verify_ssl=False) as jira_client:
        deps = IngestionDeps(jira_client=jira_client, llm_client=None, embedding_client=None)
        for ticket_id in ticket_ids:
            result = await ingest_one_ticket(ticket_id=ticket_id, deps=deps, cfg=cfg)
            observed = result.get("observed", {}) if isinstance(result, dict) else {}
            supervision = result.get("supervision", {}) if isinstance(result, dict) else {}
            metadata = observed.get("metadata", {}) if isinstance(observed, dict) else {}
            retrieval_text = str(observed.get("retrieval_text") or "").strip()
            if not retrieval_text:
                logger.warning("No retrieval_text produced from Jira ingestion for %s", ticket_id)
                continue

            payload_by_ticket[ticket_id] = {
                "retrieval_text": retrieval_text,
                "title": str(metadata.get("title") or metadata.get("summary") or ticket_id),
                "value_stream_labels": [
                    str(v)
                    for v in (supervision.get("linked_value_stream_names") or [])
                    if str(v).strip()
                ],
            }

    return payload_by_ticket


# ---------------------------------------------------------------------------
# Document conversion
# ---------------------------------------------------------------------------

def _make_document(summary_doc: Dict[str, Any]) -> Document:
    """Convert a summary dict into a LangChain Document for FAISS (V5)."""
    retrieval_text = build_retrieval_text(summary_doc)
    metadata = {
        "doc_id": summary_doc.get("doc_id", ""),
        "ticket_id": summary_doc.get("ticket_id", ""),
        "title": summary_doc.get("title", ""),
        "short_summary": summary_doc.get("short_summary", ""),
        "business_goal": summary_doc.get("business_goal", ""),
        "actors": summary_doc.get("actors", []),
        # V5: raw + canonical functions
        "direct_functions_raw": summary_doc.get("direct_functions_raw", []),
        "direct_functions_canonical": summary_doc.get("direct_functions_canonical", []),
        "implied_functions_raw": summary_doc.get("implied_functions_raw", []),
        "implied_functions_canonical": summary_doc.get("implied_functions_canonical", []),
        # Legacy compat
        "direct_functions": summary_doc.get("direct_functions", []),
        "implied_functions": summary_doc.get("implied_functions", []),
        "change_types": summary_doc.get("change_types", []),
        "domain_tags": summary_doc.get("domain_tags", []),
        "evidence_sentences": summary_doc.get("evidence_sentences", []),
        # V5: capability and operational metadata
        "capability_tags": summary_doc.get("capability_tags", []),
        "operational_footprint": summary_doc.get("operational_footprint", []),
        # Ground truth
        "value_stream_labels": summary_doc.get("value_stream_labels", []),
        "value_stream_ids": summary_doc.get("value_stream_ids", []),
        "stream_support_type": summary_doc.get("stream_support_type", {}),
        "supporting_evidence": summary_doc.get("supporting_evidence", []),
        "doc_type": "ticket_summary",
    }
    return Document(page_content=retrieval_text, metadata=metadata)


# ---------------------------------------------------------------------------
# Index operations
# ---------------------------------------------------------------------------

def build_summary_index(
    summary_docs: List[Dict[str, Any]],
    index_dir: str = DEFAULT_INDEX_DIR,
    *,
    embedding: Optional[EmbeddingService] = None,
) -> Any:
    """Build a FAISS index from a list of summary dicts and persist to disk."""
    from langchain_community.vectorstores import FAISS

    os.makedirs(index_dir, exist_ok=True)
    documents = [_make_document(doc) for doc in summary_docs]
    embeddings = embedding or get_default_embedding()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_dir)

    # Persist raw summary docs for debugging/inspection
    with open(os.path.join(index_dir, "summary_docs.json"), "w", encoding="utf-8") as f:
        json.dump(summary_docs, f, ensure_ascii=False, indent=2)

    # Persist index manifest
    manifest = {
        "schema_version": "2",          # V5 schema with richer fields
        "summary_prompt_version": "2",  # V5 prompts (raw+canonical functions)
        "retrieval_text_packing_version": "2",
        "embedding_model": getattr(embeddings, "model", "unknown"),
        "ticket_count": len(summary_docs),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(index_dir, "index_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Built FAISS summary index with %d documents at %s", len(documents), index_dir)
    return vectorstore


def load_summary_index(
    index_dir: str = DEFAULT_INDEX_DIR,
    *,
    embedding: Optional[EmbeddingService] = None,
) -> Any:
    """Load a previously persisted FAISS summary index."""
    from langchain_community.vectorstores import FAISS

    embeddings = embedding or get_default_embedding()
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def search_summary_index(
    query_text: str,
    index_dir: str = DEFAULT_INDEX_DIR,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search the summary FAISS index and return ranked analog tickets.

    Score semantics: FAISS returns L2 distance (lower = more similar). We
    normalise to a similarity score via ``1 / (1 + distance)`` so that
    **higher score always means more relevant** throughout the rest of the
    pipeline. Scores are in (0, 1].
    """
    vectorstore = load_summary_index(index_dir)
    results = vectorstore.similarity_search_with_score(query_text, k=top_k)

    out: List[Dict[str, Any]] = []
    for rank, (doc, raw_distance) in enumerate(results, start=1):
        # Normalise: higher is always better (L2 distance -> similarity)
        similarity = round(1.0 / (1.0 + float(raw_distance)), 4)
        out.append({
            "rank": rank,
            "score": similarity,
            "ticket_id": doc.metadata.get("ticket_id", ""),
            "title": doc.metadata.get("title", ""),
            "short_summary": doc.metadata.get("short_summary", ""),
            "business_goal": doc.metadata.get("business_goal", ""),
            "actors": doc.metadata.get("actors", []),
            # V5: raw + canonical functions
            "direct_functions_raw": doc.metadata.get("direct_functions_raw", []),
            "direct_functions_canonical": doc.metadata.get("direct_functions_canonical", []),
            "implied_functions_raw": doc.metadata.get("implied_functions_raw", []),
            "implied_functions_canonical": doc.metadata.get("implied_functions_canonical", []),
            # Legacy compat
            "direct_functions": doc.metadata.get("direct_functions", []),
            "implied_functions": doc.metadata.get("implied_functions", []),
            "change_types": doc.metadata.get("change_types", []),
            "domain_tags": doc.metadata.get("domain_tags", []),
            # V5 metadata
            "capability_tags": doc.metadata.get("capability_tags", []),
            "operational_footprint": doc.metadata.get("operational_footprint", []),
            "value_stream_labels": doc.metadata.get("value_stream_labels", []),
            "stream_support_type": doc.metadata.get("stream_support_type", {}),
            "supporting_evidence": doc.metadata.get("supporting_evidence", []),
            "retrieval_text": doc.page_content,
        })

    return out


# ---------------------------------------------------------------------------
# Batch ingestion: generate summaries for historical tickets -> build index
# ---------------------------------------------------------------------------

def ingest_tickets_to_index(
    ticket_ids: List[str],
    *,
    index_dir: str = DEFAULT_INDEX_DIR,
    model: str = "gpt-4o-mini",
    ticket_chunks_dir: str = "ticket_chunks",
    use_existing_summaries: bool = True,
    fetch_from_jira: bool = True,
    fallback_to_chunks: bool = True,
) -> Any:
    """
    For each ticket:
    1. Fetch ticket payload from Jira and run ingestion pipeline to produce retrieval text
    2. Read known value-stream labels from ingestion supervision output
    3. Generate semantic summary via LLM
    4. Build FAISS index from all summaries

    If Jira ingestion is unavailable and ``fallback_to_chunks`` is True,
    ticket chunk artifacts are used as a compatibility fallback.

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

    jira_payloads: Dict[str, Dict[str, Any]] = {}
    if fetch_from_jira:
        try:
            jira_payloads = _run_async(_load_ticket_inputs_from_jira(ticket_ids=ticket_ids, model=model))
            logger.info("Loaded %d/%d tickets directly from Jira", len(jira_payloads), len(ticket_ids))
        except Exception as exc:
            logger.warning("Direct Jira ingestion unavailable (%s)", exc)

    for ticket_id in ticket_ids:
        # Reuse existing summary if available
        if ticket_id in existing_by_ticket:
            logger.info("Reusing existing summary for %s", ticket_id)
            summary_docs.append(existing_by_ticket[ticket_id])
            continue

        retrieval_text = ""
        vs_labels: List[str] = []
        title = ticket_id
        jira_payload = jira_payloads.get(ticket_id)

        if jira_payload:
            retrieval_text = str(jira_payload.get("retrieval_text") or "")
            vs_labels = list(jira_payload.get("value_stream_labels") or [])
            title = str(jira_payload.get("title") or ticket_id)
        elif fallback_to_chunks:
            ticket_dir = chunks_path / ticket_id
            retrieval_text = load_ticket_retrieval_text(ticket_dir)
            vs_labels = load_ticket_vs_labels(ticket_dir)
            title = load_ticket_title(ticket_dir, ticket_id)

        if not retrieval_text:
            logger.warning("No retrieval text available for %s, skipping", ticket_id)
            continue

        # Generate summary
        try:
            summary = generate_ticket_summary(
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
