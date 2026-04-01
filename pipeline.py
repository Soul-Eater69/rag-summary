"""
Summary-first RAG pipeline orchestrator.

End-to-end flow:
  1. Clean + normalize new card text
  2. Generate semantic summary for new card
  3. Search FAISS summary index for top analog historical tickets
  4. Collect VS evidence from analog tickets
  5. Retrieve KG candidate value streams
  6. Optional: fetch raw chunks for top shortlisted tickets
  7. Build compact evidence package + LLM selection
  8. Inject canonical defaults
  9. Return structured result
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.constants import CANONICAL_VALUE_STREAMS
from core.text import clean_ppt_text, normalize_for_search

from .ingestion import generate_new_card_summary, build_retrieval_text
from .ingestion.faiss_indexer import DEFAULT_INDEX_DIR
from .retrieval import (
    retrieve_analog_tickets,
    retrieve_raw_evidence_for_tickets,
    retrieve_kg_candidates,
)
from .retrieval.summary_retriever import collect_value_stream_evidence
from .generation import select_value_streams

logger = logging.getLogger(__name__)


def run_summary_rag_pipeline(
    ppt_text: str,
    *,
    allowed_value_stream_names: Optional[List[str]] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    ticket_chunks_dir: str = "ticket_chunks",
    top_analogs: int = 5,
    top_kg_candidates: int = 20,
    include_raw_evidence: bool = True,
    max_raw_evidence_tickets: int = 3,
) -> Dict[str, Any]:
    """
    Run the summary-first RAG pipeline.

    Flow:
      1. Clean text + generate new-card summary
      2. Search FAISS for analog historical tickets
      3. Collect VS evidence from analogs
      4. Retrieve KG candidates
      5. Optional raw chunk lookup for top tickets
      6. LLM selection with compact evidence
      7. Inject canonical defaults

    Returns structured result compatible with the existing API response shape.
    """
    t_start = time.time()
    warnings: List[str] = []

    # — Step 1: Clean and summarize new card ——————————————————
    cleaned_text = clean_ppt_text(ppt_text)
    if not cleaned_text.strip():
        logger.warning("[SummaryRAG] Empty input after cleaning")
        return _empty_result("empty_input_text")

    t1 = time.time()
    try:
        new_card_summary = generate_new_card_summary(cleaned_text)
    except Exception as exc:
        logger.error("[SummaryRAG] Summary generation failed: %s", exc)
        warnings.append(f"summary_generation_failed:{type(exc).__name__}")
        # Fallback: use cleaned text as a basic summary
        new_card_summary = {
            "short_summary": cleaned_text[:300],
            "business_goal": "",
            "actors": [],
            "direct_functions": [],
            "implied_functions": [],
            "change_types": [],
            "domain_tags": [],
            "evidence_sentences": [],
        }

    logger.info("[SummaryRAG] Step 1 (summary) done in %.2fs", time.time() - t1)

    # — Step 2: Search FAISS for analog tickets ———————————————
    t2 = time.time()

    analog_tickets: List[Dict[str, Any]] = []
    try:
        analog_tickets = retrieve_analog_tickets(
            new_card_summary,
            index_dir=index_dir,
            top_k=top_analogs,
        )
    except Exception as exc:
        logger.error("[SummaryRAG] FAISS search failed: %s", exc)
        warnings.append(f"faiss_search_failed:{type(exc).__name__}")

    logger.info("[SummaryRAG] Step 2 (FAISS search) done in %.2fs | %d analogs", time.time() - t2, len(analog_tickets))

    # — Step 3: Collect VS evidence from analogs ——————————————
    t3 = time.time()
    vs_support = collect_value_stream_evidence(
        analog_tickets,
        ticket_chunks_dir=ticket_chunks_dir,
    )
    logger.info("[SummaryRAG] Step 3 (VS evidence) done in %.2fs | %d VS entries", time.time() - t3, len(vs_support))

    # — Step 4: Retrieve KG candidates ————————————————————————
    t4 = time.time()
    candidates: List[Dict[str, Any]] = []
    try:
        candidates = retrieve_kg_candidates(
            cleaned_text,
            top_k=top_kg_candidates,
            allowed_names=allowed_value_stream_names,
        )
    except Exception as exc:
        logger.error("[SummaryRAG] KG retrieval failed: %s", exc)
        warnings.append(f"kg_retrieval_failed:{type(exc).__name__}")

    logger.info("[SummaryRAG] Step 4 (KG candidates) done in %.2fs | %d candidates", time.time() - t4, len(candidates))

    # — Step 5: Optional raw evidence for top tickets —————————
    raw_evidence: List[Dict[str, Any]] = []
    if include_raw_evidence and analog_tickets:
        t5 = time.time()
        top_ticket_ids = [t["ticket_id"] for t in analog_tickets[:max_raw_evidence_tickets] if t.get("ticket_id")]
        try:
            raw_evidence = retrieve_raw_evidence_for_tickets(
                top_ticket_ids,
                ticket_chunks_dir=ticket_chunks_dir,
            )
        except Exception as exc:
            logger.warning("[SummaryRAG] Raw evidence retrieval failed: %s", exc)
        logger.info("[SummaryRAG] Step 5 (raw evidence) done in %.2fs | %d snippets", time.time() - t5, len(raw_evidence))

    # — Step 6: LLM selection —————————————————————————————————
    t6 = time.time()
    selection_result = select_value_streams(
        new_card_summary=new_card_summary,
        analog_tickets=analog_tickets,
        candidates=candidates,
        raw_evidence=raw_evidence,
        vs_support=vs_support,
        allowed_value_stream_names=allowed_value_stream_names,
    )
    logger.info("[SummaryRAG] Step 6 (LLM selection) done in %.2fs", time.time() - t6)

    # — Step 7: Inject canonical defaults —————————————————————
    canonical_output = [
        {
            "entity_id": vs.get("id", ""),
            "entity_name": vs.get("name", ""),
            "confidence": 1.0,
            "reason": "Canonical value stream (always included)",
            "category": vs.get("category", ""),
        }
        for vs in CANONICAL_VALUE_STREAMS
    ]

    # Merge: canonicals first, then LLM-selected (deduped)
    canonical_names = {(vs.get("name") or "").lower().strip() for vs in CANONICAL_VALUE_STREAMS}
    llm_selected = [
        vs for vs in selection_result.get("selected_value_streams", [])
        if (vs.get("entity_name") or "").lower().strip() not in canonical_names
    ]

    final_selected = canonical_output + llm_selected

    elapsed = time.time() - t_start
    logger.info(
        "[SummaryRAG] DONE in %.2fs | selected=%d (canonical=%d + llm=%d) | analogs=%d | warnings=%s",
        elapsed, len(final_selected), len(canonical_output), len(llm_selected),
        len(analog_tickets), warnings or "none",
    )

    return {
        "selected_value_streams": final_selected,
        "rejected_candidates": selection_result.get("rejected_candidates", []),
        "new_card_summary": new_card_summary,
        "analog_tickets": analog_tickets,
        "historical_value_stream_support": vs_support,
        "candidate_value_streams": [
            {
                "entity_id": c.get("entity_id", ""),
                "entity_name": c.get("entity_name", ""),
                "score": round(float(c.get("score") or c.get("best_score") or 0.0), 4),
                "description": (c.get("description") or "")[:220],
            }
            for c in candidates
        ],
        "raw_evidence": raw_evidence,
        "raw_response": selection_result.get("raw_response"),
        "warnings": warnings,
        "timing": {
            "total_seconds": round(elapsed, 2),
        },
    }


def _empty_result(error: str) -> Dict[str, Any]:
    return {
        "selected_value_streams": [],
        "rejected_candidates": [],
        "new_card_summary": {},
        "analog_tickets": [],
        "historical_value_stream_support": [],
        "candidate_value_streams": [],
        "raw_evidence": [],
        "raw_response": None,
        "warnings": [error],
        "error": error,
        "timing": {},
    }