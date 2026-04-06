"""
LangGraph state definition for the prediction pipeline.

Re-exports PredictionState from models.graph_state so graph modules
import from a single location.
"""

from rag_summary.models.graph_state import PredictionState

__all__ = ["PredictionState"]
