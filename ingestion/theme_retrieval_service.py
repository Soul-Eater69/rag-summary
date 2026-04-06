"""
FaissThemeRetrievalService — real ThemeRetrievalService backed by ThemeFaissIndexer (V6).

Replaces the _NoopThemeService default when a theme index has been built.

Usage:
    from summary_rag.ingestion.theme_retrieval_service import FaissThemeRetrievalService
    from summary_rag.ingestion.adapters import get_default_embedding

    theme_svc = FaissThemeRetrievalService(
        theme_index_dir="config/theme_index",
        embedding_svc=get_default_embedding(),
    )
    result = run_summary_rag_pipeline(ppt_text, theme_svc=theme_svc)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from summary_rag.ingestion.theme_indexer import ThemeFaissIndexer
from summary_rag.ingestion.adapters import EmbeddingService

logger = logging.getLogger(__name__)


class FaissThemeRetrievalService:
    """
    Implements ThemeRetrievalService Protocol using ThemeFaissIndexer.

    Score formula per (theme, vs_name):
        raw_sim = FAISS cosine similarity (0–1)
        vs_fraction = vs_support_fractions[vs_name]  (how often VS appears in theme)
        quality_factor = min(1.0, 0.5 + 0.5 * cohesion_score)
        candidate_score = raw_sim * vs_fraction * quality_factor

    This caps theme evidence below 0.75 in practice (theme-only = pattern_inferred).
    """

    def __init__(
        self,
        theme_index_dir: str,
        embedding_svc: EmbeddingService,
        *,
        min_vs_support_fraction: float = 0.30,
    ) -> None:
        self._indexer = ThemeFaissIndexer(theme_index_dir)
        self._embedding_svc = embedding_svc
        self._min_vs_support_fraction = min_vs_support_fraction

    def retrieve_theme_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        allowed_names: Optional[List[str]] = None,
        cutoff_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed query_text, search theme index, expand each theme into per-VS candidates.

        Returns list of candidate dicts (one per VS name per theme hit):
          candidate_name: str
          source: "theme"
          score: float (calibrated 0–0.75)
          theme_id: str
          theme_label: str
          vs_support_fraction: float
          cue_phrases: List[str]
          capability_tags: List[str]
          member_count: int
          evidence_snippets: List[str]  (top-3 cue phrases as text)
          theme_match_count: int         (always 1; incremented by _inject_footprint_patterns)
        """
        if not query_text or not query_text.strip():
            return []

        # Embed the query
        try:
            embedding = self._embedding_svc.embed_query(query_text)
        except Exception as exc:
            logger.warning("[FaissThemeRetrievalService] Embedding failed: %s", exc)
            return []

        # Search theme index
        theme_hits = self._indexer.search(
            embedding,
            top_k=top_k,
            allowed_vs_names=allowed_names,
            cutoff_date=cutoff_date,
        )

        # Expand: one candidate per VS in each theme hit
        candidates: Dict[str, Dict[str, Any]] = {}  # candidate_name → best candidate

        for hit in theme_hits:
            sim = hit.get("similarity_score", 0.0)
            cohesion = hit.get("cohesion_score", 0.5)
            quality_factor = min(1.0, 0.5 + 0.5 * cohesion)

            fractions: Dict[str, float] = hit.get("vs_support_fractions", {})

            for vs_name in hit.get("value_stream_names", []):
                fraction = fractions.get(vs_name, 0.0)
                if fraction < self._min_vs_support_fraction:
                    continue

                score = round(sim * fraction * quality_factor, 4)

                cues = hit.get("cue_phrases", [])
                evidence_snippets = [
                    f"theme:{hit['theme_label']}, cue:{c}" for c in cues[:3]
                ]

                # Keep highest-scoring theme hit per VS
                existing = candidates.get(vs_name)
                if existing is None or score > existing.get("score", 0.0):
                    candidates[vs_name] = {
                        "entity_name": vs_name,
                        "candidate_name": vs_name,
                        "source": "theme",
                        "score": score,
                        "theme_id": hit.get("theme_id", ""),
                        "theme_label": hit.get("theme_label", ""),
                        "vs_support_fraction": fraction,
                        "cue_phrases": cues[:5],
                        "capability_tags": hit.get("capability_tags", []),
                        "member_count": hit.get("member_count", 0),
                        "evidence_snippets": evidence_snippets,
                        "snippets": evidence_snippets,  # also exposed as snippets for _add_source
                        "theme_match_count": 1,
                    }

        result = sorted(candidates.values(), key=lambda c: -c["score"])
        logger.info(
            "[FaissThemeRetrievalService] %d theme hits → %d VS candidates",
            len(theme_hits), len(result),
        )
        return result
