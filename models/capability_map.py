"""
Pydantic CapabilityMap models (V5 architecture).

Replaces the raw dict parsing in generation/capability_mapper.py.
Loaded from config/capability_map.yaml.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class CapabilityCluster(BaseModel):
    """A single capability cluster mapping cues to value streams."""

    description: str = ""
    direct_cues: List[str] = Field(default_factory=list)
    indirect_cues: List[str] = Field(default_factory=list)
    canonical_functions: List[str] = Field(default_factory=list)
    promote_value_streams: List[str] = Field(default_factory=list)
    related_value_streams: List[str] = Field(default_factory=list)
    weight: float = 1.0
    min_signal_strength: float = 0.5

    class Config:
        extra = "allow"


class CapabilityMap(BaseModel):
    """Full capability map loaded from YAML config."""

    version: int = 1
    capabilities: Dict[str, CapabilityCluster] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "CapabilityMap":
        """Parse from raw YAML payload."""
        version = int(payload.get("version", 1))
        raw_caps = payload.get("capabilities") or {}
        capabilities: Dict[str, CapabilityCluster] = {}
        for name, cluster in raw_caps.items():
            if not isinstance(cluster, dict):
                continue
            # Support both promote_value_streams and promoted_value_streams
            promote = cluster.get("promote_value_streams") or cluster.get("promoted_value_streams") or []
            if not promote:
                continue
            capabilities[name] = CapabilityCluster(
                description=cluster.get("description", ""),
                direct_cues=list(cluster.get("direct_cues") or []),
                indirect_cues=list(cluster.get("indirect_cues") or []),
                canonical_functions=list(cluster.get("canonical_functions") or []),
                promote_value_streams=list(promote),
                related_value_streams=list(cluster.get("related_value_streams") or []),
                weight=float(cluster.get("weight") or 1.0),
                min_signal_strength=float(cluster.get("min_signal_strength") or 0.5),
            )
        return cls(version=version, capabilities=capabilities)
