"""
Concrete adapter implementations (V6).

Wraps the internal FaissIndexer and ticket_chunks filesystem reader behind
the SummaryIndexService and RawEvidenceService Protocols defined in adapters.py.

These allow nodes and the ServiceContainer to work with typed interfaces
instead of directly importing the concrete classes, making test injection easy.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FaissIndexAdapter:
    """
    Wraps FaissIndexer behind the SummaryIndexService Protocol.

    Searches the FAISS summary index for analog tickets given a query text.
    Embeds the query using embedding_svc, then delegates to FaissIndexer.search().
    """

    def __init__(
        self,
        index_dir: str,
        embedding_svc,
    ) -> None:
        self._index_dir = index_dir
        self._embedding_svc = embedding_svc
        self._indexer = None  # lazy-loaded

    def _get_indexer(self):
        if self._indexer is None:
            from summary_rag.ingestion.faiss_indexer import FaissIndexer
            self._indexer = FaissIndexer(self._index_dir)
        return self._indexer

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        allowed_vs_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Embed query_text and search the FAISS summary index.

        Returns list of analog ticket dicts with at least:
          ticket_id, score, value_stream_labels, retrieval_text
        """
        if not query_text or not query_text.strip():
            return []
        try:
            embedding = self._embedding_svc.embed_query(query_text)
        except Exception as exc:
            logger.warning("[FaissIndexAdapter] Embedding failed: %s", exc)
            return []
        try:
            indexer = self._get_indexer()
            return indexer.search(
                embedding,
                top_k=top_k,
                allowed_vs_names=allowed_vs_names,
            )
        except Exception as exc:
            logger.warning("[FaissIndexAdapter] Search failed: %s", exc)
            return []


class TicketChunksAdapter:
    """
    Wraps the filesystem ticket_chunks/ directory behind the RawEvidenceService Protocol.

    Reads pre-chunked text files for a given ticket_id from disk.
    Each chunk file is expected to be JSON with at least {"text": str, "score": float}.
    """

    def __init__(self, ticket_chunks_dir: str) -> None:
        self._ticket_chunks_dir = ticket_chunks_dir

    def get_chunks_for_ticket(
        self,
        ticket_id: str,
        *,
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> List[dict]:
        """
        Return up to top_k chunk dicts for a ticket.

        Looks for files matching ticket_chunks/<ticket_id>.json or
        ticket_chunks/<ticket_id>/*.json.

        Returns list of chunk dicts with at least {"text": str, "score": float}.
        """
        if not ticket_id:
            return []

        chunks: List[Dict[str, Any]] = []

        # Single-file pattern: ticket_chunks/<ticket_id>.json
        single_path = os.path.join(self._ticket_chunks_dir, f"{ticket_id}.json")
        if os.path.exists(single_path):
            chunks = self._load_json_file(single_path)

        # Directory pattern: ticket_chunks/<ticket_id>/*.json
        ticket_dir = os.path.join(self._ticket_chunks_dir, ticket_id)
        if not chunks and os.path.isdir(ticket_dir):
            for fname in sorted(os.listdir(ticket_dir)):
                if fname.endswith(".json"):
                    chunks.extend(self._load_json_file(os.path.join(ticket_dir, fname)))
                    if len(chunks) >= top_k * 2:
                        break

        # Sort by score descending, return top_k
        chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        return chunks[:top_k]

    def _load_json_file(self, path: str) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return []
        except Exception as exc:
            logger.warning("[TicketChunksAdapter] Failed to load %s: %s", path, exc)
            return []
