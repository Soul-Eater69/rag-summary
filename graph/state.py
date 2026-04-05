"""
LangGraph state definition for the prediction pipeline.

Re-exports PredictionState from models.graph_state so graph modules
import from a single location.
"""

from summary_rag.models.graph_state import PredictionState

__all__ = ["PredictionState"]
