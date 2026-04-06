"""
LangGraph StateGraph assembly for the V5 prediction pipeline.

Builds the graph once (or per-call if stateless) and exposes run_prediction_graph()
as the primary callable entry point.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from summary_rag.models.graph_state import PredictionState
from summary_rag.ingestion.faiss_indexer import DEFAULT_INDEX_DIR
from .nodes import (
    node_clean_and_summarize,
    node_retrieve_analogs,
    node_collect_vs_evidence,
    node_retrieve_kg,
    node_retrieve_themes,
    node_map_capabilities,
    node_extract_card_candidates,
    node_collect_raw_evidence,
    node_build_evidence,
    node_fuse_scores,
    node_verify_candidates,
    node_finalize_selection,
    node_finalize_output,
)
from .edges import (
    should_continue_after_summarize,
    should_continue_after_analogs,
    should_run_finalize,
)

logger = logging.getLogger(__name__)


def build_prediction_graph():
    """
    Construct and return the compiled LangGraph StateGraph.

    Requires langgraph to be installed. Raises ImportError with a helpful
    message if langgraph is not available.
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        raise ImportError(
            "langgraph is required for graph-based orchestration. "
            "Install it with: pip install langgraph"
        ) from exc

    graph = StateGraph(PredictionState)

    # Register all nodes
    graph.add_node("clean_and_summarize", node_clean_and_summarize)
    graph.add_node("retrieve_analogs", node_retrieve_analogs)
    graph.add_node("collect_vs_evidence", node_collect_vs_evidence)
    graph.add_node("retrieve_kg", node_retrieve_kg)
    graph.add_node("retrieve_themes", node_retrieve_themes)
    graph.add_node("map_capabilities", node_map_capabilities)
    graph.add_node("extract_card_candidates", node_extract_card_candidates)
    graph.add_node("collect_raw_evidence", node_collect_raw_evidence)
    graph.add_node("build_evidence", node_build_evidence)
    graph.add_node("fuse_scores", node_fuse_scores)
    graph.add_node("verify_candidates", node_verify_candidates)
    graph.add_node("finalize_selection", node_finalize_selection)
    graph.add_node("finalize_output", node_finalize_output)

    # Entry point
    graph.set_entry_point("clean_and_summarize")

    # Conditional: abort if empty input
    graph.add_conditional_edges(
        "clean_and_summarize",
        should_continue_after_summarize,
        {
            "retrieve_analogs": "retrieve_analogs",
            "end": END,
        },
    )

    # Conditional: skip VS evidence if no analogs
    graph.add_conditional_edges(
        "retrieve_analogs",
        should_continue_after_analogs,
        {
            "collect_vs_evidence": "collect_vs_evidence",
            "retrieve_kg": "retrieve_kg",
        },
    )

    # VS evidence always feeds into KG retrieval
    graph.add_edge("collect_vs_evidence", "retrieve_kg")

    # Sequential pipeline after KG
    graph.add_edge("retrieve_kg", "retrieve_themes")
    graph.add_edge("retrieve_themes", "map_capabilities")
    graph.add_edge("map_capabilities", "extract_card_candidates")
    graph.add_edge("extract_card_candidates", "collect_raw_evidence")
    graph.add_edge("collect_raw_evidence", "build_evidence")
    graph.add_edge("build_evidence", "fuse_scores")
    graph.add_edge("fuse_scores", "verify_candidates")

    # Conditional: skip pass-2 if pass-1 was empty
    graph.add_conditional_edges(
        "verify_candidates",
        should_run_finalize,
        {
            "finalize_selection": "finalize_selection",
            "finalize_output": "finalize_output",
        },
    )

    graph.add_edge("finalize_selection", "finalize_output")
    graph.add_edge("finalize_output", END)

    return graph.compile()


def run_prediction_graph(
    ppt_text: str,
    *,
    allowed_value_stream_names: Optional[List[str]] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    ticket_chunks_dir: str = "ticket_chunks",
    top_analogs: int = 5,
    top_kg_candidates: int = 20,
    include_raw_evidence: bool = True,
    max_raw_evidence_tickets: int = 3,
    min_candidate_floor: int = 8,
    llm=None,
    theme_svc=None,
    intake_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the V5 prediction pipeline via LangGraph.

    This is the primary entry point for graph-based execution. Pipeline config
    is injected into state as private underscore keys that nodes read from.

    Returns the final PredictionState dict with all pipeline artifacts.
    """
    t_start = time.time()

    # Build initial state
    initial_state: PredictionState = {
        "raw_text": ppt_text,
        "allowed_value_stream_names": allowed_value_stream_names or [],
        "top_k_analogs": top_analogs,
        "errors": [],
        "warnings": [],
        "timing": {},
        # Pipeline config passed as private state keys (read by nodes)
        "_index_dir": index_dir,  # type: ignore[typeddict-unknown-key]
        "_ticket_chunks_dir": ticket_chunks_dir,  # type: ignore[typeddict-unknown-key]
        "_top_kg_candidates": top_kg_candidates,  # type: ignore[typeddict-unknown-key]
        "_include_raw_evidence": include_raw_evidence,  # type: ignore[typeddict-unknown-key]
        "_max_raw_evidence_tickets": max_raw_evidence_tickets,  # type: ignore[typeddict-unknown-key]
        "_min_candidate_floor": min_candidate_floor,  # type: ignore[typeddict-unknown-key]
        "_llm": llm,  # type: ignore[typeddict-unknown-key]
        "_theme_svc": theme_svc,  # type: ignore[typeddict-unknown-key]
        "_intake_date": intake_date,  # type: ignore[typeddict-unknown-key]
    }

    try:
        app = build_prediction_graph()
        final_state = app.invoke(initial_state)
    except ImportError:
        # LangGraph not installed — fall back to sequential imperative execution
        logger.warning(
            "[run_prediction_graph] langgraph not available, running sequential fallback"
        )
        final_state = _run_sequential(initial_state)

    elapsed = time.time() - t_start
    final_state["timing"] = {"total_seconds": round(elapsed, 2)}

    logger.info(
        "[run_prediction_graph] DONE in %.2fs | direct=%d | pattern=%d | no_evidence=%d",
        elapsed,
        len(final_state.get("directly_supported", [])),
        len(final_state.get("pattern_inferred", [])),
        len(final_state.get("no_evidence", [])),
    )
    return dict(final_state)


def _run_sequential(state: PredictionState) -> PredictionState:
    """
    Sequential fallback when LangGraph is not installed.

    Runs each node in order, merging returned dicts into state.
    Respects the same conditional logic as the graph edges.
    """
    from .edges import (
        should_continue_after_summarize,
        should_continue_after_analogs,
        should_run_finalize,
    )

    def _merge(s: PredictionState, updates: Dict[str, Any]) -> PredictionState:
        return {**s, **updates}  # type: ignore[return-value]

    state = _merge(state, node_clean_and_summarize(state))
    if should_continue_after_summarize(state) == "end":
        return state

    state = _merge(state, node_retrieve_analogs(state))
    if should_continue_after_analogs(state) == "collect_vs_evidence":
        state = _merge(state, node_collect_vs_evidence(state))

    state = _merge(state, node_retrieve_kg(state))
    state = _merge(state, node_retrieve_themes(state))
    state = _merge(state, node_map_capabilities(state))
    state = _merge(state, node_extract_card_candidates(state))
    state = _merge(state, node_collect_raw_evidence(state))
    state = _merge(state, node_build_evidence(state))
    state = _merge(state, node_fuse_scores(state))
    state = _merge(state, node_verify_candidates(state))

    if should_run_finalize(state) == "finalize_selection":
        state = _merge(state, node_finalize_selection(state))

    state = _merge(state, node_finalize_output(state))
    return state
