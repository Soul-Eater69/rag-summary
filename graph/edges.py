"""
LangGraph conditional edge logic for the prediction pipeline.

Edges inspect the current state and route to the appropriate next node.
"""

from __future__ import annotations

from typing import Literal

from rag_summary.models.graph_state import PredictionState


def should_continue_after_summarize(
    state: PredictionState,
) -> Literal["retrieve_analogs", "end"]:
    """
    After clean_and_summarize: abort if input was empty.
    """
    errors = state.get("errors", [])
    if "empty_input_text" in errors:
        return "end"
    return "retrieve_analogs"


def should_continue_after_analogs(
    state: PredictionState,
) -> Literal["collect_vs_evidence", "retrieve_kg"]:
    """
    After retrieve_analogs: skip VS evidence collection if no analogs found
    (jump straight to KG retrieval, which always runs).
    """
    analog_tickets = state.get("analog_tickets", [])
    if not analog_tickets:
        return "retrieve_kg"
    return "collect_vs_evidence"


def should_run_finalize(
    state: PredictionState,
) -> Literal["finalize_selection", "finalize_output"]:
    """
    After verify_candidates: run pass-2 finalize only if pass-1 produced
    judgments. If judgment list is empty (total LLM failure), skip to output.
    """
    judgments = state.get("verify_judgments") or []
    if judgments:
        return "finalize_selection"
    return "finalize_output"
