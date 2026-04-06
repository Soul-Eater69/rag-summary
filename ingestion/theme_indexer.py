"""
ThemeFaissIndexer — loads and searches the offline theme FAISS index (V6).

Built by tools/build_theme_index.py. Searched at runtime by
ingestion/theme_retrieval_service.py.

Index layout (all files in theme_index_dir/):
  theme_index.faiss   — FAISS flat L2 index over theme centroid embeddings
  theme_docs.json     — List[ThemeDoc] serialized (without embeddings)
  theme_manifest.json — ThemeIndexManifest with build metadata
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ThemeFaissIndexer:
    """
    Loads and queries the theme FAISS index.

    Wraps faiss + numpy. These are already required by the existing
    FaissIndexer, so no new mandatory deps are added.
    """

    def __init__(self, theme_index_dir: str) -> None:
        self._dir = theme_index_dir
        self._index = None          # faiss index (loaded lazily)
        self._theme_docs: List[Dict[str, Any]] = []
        self._loaded = False

    def _load(self) -> None:
        """Lazy load — called on first search()."""
        if self._loaded:
            return
        index_path = os.path.join(self._dir, "theme_index.faiss")
        docs_path = os.path.join(self._dir, "theme_docs.json")

        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            logger.warning(
                "[ThemeFaissIndexer] Index not found at %s — theme source inactive. "
                "Run tools/build_theme_index.py to build.", self._dir
            )
            self._loaded = True
            return

        try:
            import faiss  # type: ignore[import]
            import numpy as np

            self._index = faiss.read_index(index_path)
            with open(docs_path, "r", encoding="utf-8") as f:
                self._theme_docs = json.load(f)
            logger.info(
                "[ThemeFaissIndexer] Loaded %d themes from %s",
                len(self._theme_docs), self._dir,
            )
        except ImportError:
            logger.warning("[ThemeFaissIndexer] faiss not installed; theme source inactive")
        except Exception as exc:
            logger.warning("[ThemeFaissIndexer] Load failed: %s", exc)

        self._loaded = True

    def search(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 10,
        allowed_vs_names: Optional[List[str]] = None,
        cutoff_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top_k theme matches for a query embedding.

        Each result dict:
          theme_id, theme_label, theme_description, value_stream_names,
          vs_support_fractions, cue_phrases, member_count,
          cohesion_score, similarity_score, last_ticket_ingested_at
        """
        self._load()
        if self._index is None or not self._theme_docs:
            return []

        try:
            import numpy as np
            q = np.array([query_embedding], dtype="float32")
            # Normalise for cosine similarity via L2 search on unit vectors
            faiss_norm = np.linalg.norm(q)
            if faiss_norm > 0:
                q /= faiss_norm

            k = min(top_k * 3, len(self._theme_docs))  # over-fetch, filter after
            distances, indices = self._index.search(q, k)
        except Exception as exc:
            logger.warning("[ThemeFaissIndexer] Search failed: %s", exc)
            return []

        results = []
        seen_ids = set()
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._theme_docs):
                continue
            doc = self._theme_docs[idx]
            theme_id = doc.get("theme_id", "")
            if theme_id in seen_ids:
                continue
            seen_ids.add(theme_id)

            # Cutoff: skip themes whose most recent ticket is after intake date
            if cutoff_date:
                last_ingested = doc.get("last_ticket_ingested_at", "")
                if last_ingested and last_ingested > cutoff_date:
                    continue

            # Cosine similarity from L2 distance on normalised vectors:
            # sim = 1 - dist/2  (since ||a-b||^2 = 2 - 2*cos for unit vectors)
            similarity = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))

            # Filter VS by allowed_names
            vs_names = doc.get("value_stream_names", [])
            allowed_set = set(allowed_vs_names) if allowed_vs_names else None
            if allowed_set:
                vs_names = [n for n in vs_names if n in allowed_set]
            if not vs_names:
                continue

            results.append({
                "theme_id": theme_id,
                "theme_label": doc.get("theme_label", ""),
                "theme_description": doc.get("theme_description", ""),
                "value_stream_names": vs_names,
                "vs_support_fractions": doc.get("vs_support_fractions", {}),
                "cue_phrases": doc.get("cue_phrases", []),
                "canonical_functions": doc.get("canonical_functions", []),
                "capability_tags": doc.get("capability_tags", []),
                "member_count": doc.get("member_count", 0),
                "cohesion_score": doc.get("cohesion_score", 0.0),
                "similarity_score": round(similarity, 4),
                "last_ticket_ingested_at": doc.get("last_ticket_ingested_at", ""),
            })

            if len(results) >= top_k:
                break

        return results
