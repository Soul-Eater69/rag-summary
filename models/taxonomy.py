"""
Taxonomy registry models (V1).

TaxonomyStream: single value-stream entry with aliases, family, overlap,
and preference/suppression rules.

TaxonomyRegistry: collection of all streams plus pre-built lookup indices
(canonical_label_map, family_index, overlap_index).

Used by:
  - taxonomy/registry_loader.py  → loads YAML into TaxonomyRegistry
  - (Phase 4) taxonomy/policy_reranker.py  → applies overlap/suppression rules
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, model_validator


class TaxonomyStream(BaseModel):
    """A single canonical value-stream definition."""

    id: int
    canonical_name: str
    evaluation_name: str = ""
    aliases: List[str] = Field(default_factory=list)
    family: str = ""
    scope: str = ""
    broad: bool = False
    overlaps_with: List[str] = Field(default_factory=list)
    preferred_over: List[str] = Field(default_factory=list)
    suppress_if_preferred: bool = False

    @model_validator(mode="after")
    def _default_eval_name(self) -> "TaxonomyStream":
        if not self.evaluation_name:
            self.evaluation_name = self.canonical_name
        return self


class TaxonomyRegistry(BaseModel):
    """
    Full taxonomy registry with pre-built lookup indices.

    Constructed by taxonomy/registry_loader.py from config/taxonomy_registry.yaml.
    """

    version: str = "1"
    streams: List[TaxonomyStream] = Field(default_factory=list)

    # ----- pre-built indices (populated by build_indices) -----
    canonical_label_map: Dict[str, str] = Field(default_factory=dict)
    """Maps every known alias (lowercased) → canonical_name."""

    family_index: Dict[str, List[str]] = Field(default_factory=dict)
    """Maps family label → list of canonical_names in that family."""

    overlap_index: Dict[str, List[str]] = Field(default_factory=dict)
    """Maps canonical_name → list of canonical_names it overlaps with."""

    preferred_index: Dict[str, List[str]] = Field(default_factory=dict)
    """Maps canonical_name → list of canonical_names it is preferred over."""

    suppressible: Set[str] = Field(default_factory=set)
    """Set of canonical_names that should be suppressed when a preferred sibling is present."""

    broad_streams: Set[str] = Field(default_factory=set)
    """Set of canonical_names flagged as broad umbrella concepts."""

    def build_indices(self) -> "TaxonomyRegistry":
        """
        Populate all lookup indices from the streams list.

        Called once after loading. Returns self for chaining.
        """
        self.canonical_label_map.clear()
        self.family_index.clear()
        self.overlap_index.clear()
        self.preferred_index.clear()
        self.suppressible.clear()
        self.broad_streams.clear()

        for stream in self.streams:
            canon = stream.canonical_name

            # canonical_label_map: canonical name + all aliases → canonical
            self.canonical_label_map[canon.lower()] = canon
            for alias in stream.aliases:
                self.canonical_label_map[alias.lower()] = canon

            # family_index
            if stream.family:
                self.family_index.setdefault(stream.family, []).append(canon)

            # overlap_index
            if stream.overlaps_with:
                self.overlap_index[canon] = list(stream.overlaps_with)

            # preferred_index
            if stream.preferred_over:
                self.preferred_index[canon] = list(stream.preferred_over)

            # suppressible
            if stream.suppress_if_preferred:
                self.suppressible.add(canon)

            # broad_streams
            if stream.broad:
                self.broad_streams.add(canon)

        return self

    def canonicalize(self, name: str) -> str:
        """
        Map any alias/variant to the canonical name.

        Returns the original string unchanged if no match is found.
        """
        return self.canonical_label_map.get(name.lower(), name)

    def get_stream(self, name: str) -> Optional[TaxonomyStream]:
        """Look up a TaxonomyStream by canonical name or alias."""
        canon = self.canonicalize(name)
        for s in self.streams:
            if s.canonical_name == canon:
                return s
        return None

    def get_family(self, name: str) -> Optional[str]:
        """Return the family label for a given stream name/alias."""
        stream = self.get_stream(name)
        return stream.family if stream else None

    def get_overlaps(self, name: str) -> List[str]:
        """Return canonical_names that overlap with the given stream."""
        canon = self.canonicalize(name)
        return self.overlap_index.get(canon, [])

    def is_broad(self, name: str) -> bool:
        """Check if a stream is flagged as a broad umbrella concept."""
        return self.canonicalize(name) in self.broad_streams

    def should_suppress(self, name: str) -> bool:
        """Check if a stream should be suppressed when preferred siblings are present."""
        return self.canonicalize(name) in self.suppressible
