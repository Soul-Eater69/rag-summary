"""
Tests for Phase 1 — taxonomy registry loading, model construction,
alias resolution, and error handling.
"""

from __future__ import annotations

import os
import pathlib
import textwrap
import tempfile

import pytest
import yaml

from rag_summary.models.taxonomy import TaxonomyStream, TaxonomyRegistry
from rag_summary.taxonomy.registry_loader import (
    load_taxonomy_registry,
    try_load_taxonomy_registry,
    TaxonomyRegistryError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_REGISTRY_YAML = textwrap.dedent("""\
    version: "1"
    streams:
      - id: 1
        canonical_name: "Order to Cash"
        aliases:
          - "O2C"
          - "order to cash"
        family: "finance"
        scope: "Revenue cycle"
        broad: true
        overlaps_with:
          - "Issue Payment"
        preferred_over: []
        suppress_if_preferred: false

      - id: 2
        canonical_name: "Issue Payment"
        aliases:
          - "Payment Issuance"
        family: "finance"
        scope: "Payment disbursement"
        broad: false
        overlaps_with:
          - "Order to Cash"
        preferred_over: []
        suppress_if_preferred: false

      - id: 3
        canonical_name: "Promote Community Health"
        aliases:
          - "Community Health"
        family: "care_management"
        scope: "Population health outreach"
        broad: true
        overlaps_with:
          - "Participate in Health Management Program"
        preferred_over: []
        suppress_if_preferred: true

      - id: 4
        canonical_name: "Participate in Health Management Program"
        aliases:
          - "HMP"
        family: "care_management"
        scope: "Disease/care management programs"
        broad: false
        overlaps_with:
          - "Promote Community Health"
        preferred_over:
          - "Promote Community Health"
        suppress_if_preferred: false
""")


@pytest.fixture
def minimal_yaml_path(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "taxonomy_registry.yaml"
    p.write_text(_MINIMAL_REGISTRY_YAML, encoding="utf-8")
    return p


@pytest.fixture
def minimal_registry(minimal_yaml_path: pathlib.Path) -> TaxonomyRegistry:
    return load_taxonomy_registry(minimal_yaml_path)


# ---------------------------------------------------------------------------
# TaxonomyStream model tests
# ---------------------------------------------------------------------------

class TestTaxonomyStream:

    def test_defaults(self):
        s = TaxonomyStream(id=1, canonical_name="Test")
        assert s.evaluation_name == "Test"  # auto-populated
        assert s.aliases == []
        assert s.family == ""
        assert s.broad is False
        assert s.suppress_if_preferred is False

    def test_explicit_evaluation_name(self):
        s = TaxonomyStream(id=1, canonical_name="Test", evaluation_name="TestEval")
        assert s.evaluation_name == "TestEval"


# ---------------------------------------------------------------------------
# TaxonomyRegistry model tests
# ---------------------------------------------------------------------------

class TestTaxonomyRegistry:

    def test_build_indices_canonical_label_map(self, minimal_registry: TaxonomyRegistry):
        m = minimal_registry.canonical_label_map
        assert m["order to cash"] == "Order to Cash"
        assert m["o2c"] == "Order to Cash"
        assert m["payment issuance"] == "Issue Payment"
        assert m["hmp"] == "Participate in Health Management Program"
        assert m["community health"] == "Promote Community Health"

    def test_build_indices_family(self, minimal_registry: TaxonomyRegistry):
        assert "finance" in minimal_registry.family_index
        assert "Order to Cash" in minimal_registry.family_index["finance"]
        assert "Issue Payment" in minimal_registry.family_index["finance"]
        assert "care_management" in minimal_registry.family_index

    def test_build_indices_overlap(self, minimal_registry: TaxonomyRegistry):
        assert "Issue Payment" in minimal_registry.overlap_index.get("Order to Cash", [])
        assert "Order to Cash" in minimal_registry.overlap_index.get("Issue Payment", [])

    def test_build_indices_preferred(self, minimal_registry: TaxonomyRegistry):
        # HMP is preferred over Promote Community Health
        assert "Promote Community Health" in minimal_registry.preferred_index.get(
            "Participate in Health Management Program", []
        )

    def test_build_indices_suppressible(self, minimal_registry: TaxonomyRegistry):
        assert "Promote Community Health" in minimal_registry.suppressible
        assert "Order to Cash" not in minimal_registry.suppressible

    def test_build_indices_broad(self, minimal_registry: TaxonomyRegistry):
        assert "Order to Cash" in minimal_registry.broad_streams
        assert "Issue Payment" not in minimal_registry.broad_streams

    def test_canonicalize_exact(self, minimal_registry: TaxonomyRegistry):
        assert minimal_registry.canonicalize("Order to Cash") == "Order to Cash"

    def test_canonicalize_alias(self, minimal_registry: TaxonomyRegistry):
        assert minimal_registry.canonicalize("O2C") == "Order to Cash"
        assert minimal_registry.canonicalize("o2c") == "Order to Cash"

    def test_canonicalize_unknown_passthrough(self, minimal_registry: TaxonomyRegistry):
        assert minimal_registry.canonicalize("Unknown Stream") == "Unknown Stream"

    def test_get_stream(self, minimal_registry: TaxonomyRegistry):
        s = minimal_registry.get_stream("O2C")
        assert s is not None
        assert s.canonical_name == "Order to Cash"
        assert s.family == "finance"

    def test_get_stream_unknown(self, minimal_registry: TaxonomyRegistry):
        assert minimal_registry.get_stream("No Such Stream") is None

    def test_get_family(self, minimal_registry: TaxonomyRegistry):
        assert minimal_registry.get_family("O2C") == "finance"
        assert minimal_registry.get_family("HMP") == "care_management"
        assert minimal_registry.get_family("Unknown") is None

    def test_get_overlaps(self, minimal_registry: TaxonomyRegistry):
        overlaps = minimal_registry.get_overlaps("Order to Cash")
        assert "Issue Payment" in overlaps

    def test_is_broad(self, minimal_registry: TaxonomyRegistry):
        assert minimal_registry.is_broad("Order to Cash") is True
        assert minimal_registry.is_broad("Issue Payment") is False

    def test_should_suppress(self, minimal_registry: TaxonomyRegistry):
        assert minimal_registry.should_suppress("Promote Community Health") is True
        assert minimal_registry.should_suppress("Issue Payment") is False


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestRegistryLoader:

    def test_load_success(self, minimal_yaml_path: pathlib.Path):
        reg = load_taxonomy_registry(minimal_yaml_path)
        assert len(reg.streams) == 4
        assert reg.version == "1"
        # Indices are populated
        assert len(reg.canonical_label_map) > 0

    def test_load_default_path(self):
        """Loading from default config path should work if the file exists."""
        default_path = (
            pathlib.Path(__file__).resolve().parent.parent / "config" / "taxonomy_registry.yaml"
        )
        if default_path.exists():
            reg = load_taxonomy_registry()
            assert len(reg.streams) > 0
        else:
            pytest.skip("Default taxonomy_registry.yaml not found")

    def test_missing_file_raises(self, tmp_path: pathlib.Path):
        with pytest.raises(TaxonomyRegistryError, match="not found"):
            load_taxonomy_registry(tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_raises(self, tmp_path: pathlib.Path):
        p = tmp_path / "bad.yaml"
        p.write_text(":\n  - :\n    bad: [unclosed", encoding="utf-8")
        with pytest.raises(TaxonomyRegistryError, match="parse"):
            load_taxonomy_registry(p)

    def test_not_a_mapping_raises(self, tmp_path: pathlib.Path):
        p = tmp_path / "list.yaml"
        p.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(TaxonomyRegistryError, match="mapping"):
            load_taxonomy_registry(p)

    def test_missing_streams_key_raises(self, tmp_path: pathlib.Path):
        p = tmp_path / "no_streams.yaml"
        p.write_text("version: '1'\n", encoding="utf-8")
        with pytest.raises(TaxonomyRegistryError, match="'streams'"):
            load_taxonomy_registry(p)

    def test_stream_missing_id_raises(self, tmp_path: pathlib.Path):
        content = textwrap.dedent("""\
            version: "1"
            streams:
              - canonical_name: "Test"
        """)
        p = tmp_path / "no_id.yaml"
        p.write_text(content, encoding="utf-8")
        with pytest.raises(TaxonomyRegistryError, match="'id'"):
            load_taxonomy_registry(p)

    def test_stream_missing_canonical_name_raises(self, tmp_path: pathlib.Path):
        content = textwrap.dedent("""\
            version: "1"
            streams:
              - id: 1
        """)
        p = tmp_path / "no_name.yaml"
        p.write_text(content, encoding="utf-8")
        with pytest.raises(TaxonomyRegistryError, match="'canonical_name'"):
            load_taxonomy_registry(p)

    def test_duplicate_id_raises(self, tmp_path: pathlib.Path):
        content = textwrap.dedent("""\
            version: "1"
            streams:
              - id: 1
                canonical_name: "A"
              - id: 1
                canonical_name: "B"
        """)
        p = tmp_path / "dup_id.yaml"
        p.write_text(content, encoding="utf-8")
        with pytest.raises(TaxonomyRegistryError, match="Duplicate stream id"):
            load_taxonomy_registry(p)

    def test_duplicate_canonical_name_raises(self, tmp_path: pathlib.Path):
        content = textwrap.dedent("""\
            version: "1"
            streams:
              - id: 1
                canonical_name: "Same Name"
              - id: 2
                canonical_name: "Same Name"
        """)
        p = tmp_path / "dup_name.yaml"
        p.write_text(content, encoding="utf-8")
        with pytest.raises(TaxonomyRegistryError, match="Duplicate canonical_name"):
            load_taxonomy_registry(p)

    def test_try_load_returns_none_on_missing(self, tmp_path: pathlib.Path):
        result = try_load_taxonomy_registry(tmp_path / "nonexistent.yaml")
        assert result is None

    def test_try_load_returns_registry_on_success(self, minimal_yaml_path: pathlib.Path):
        result = try_load_taxonomy_registry(minimal_yaml_path)
        assert result is not None
        assert len(result.streams) == 4

    def test_empty_streams_list(self, tmp_path: pathlib.Path):
        content = textwrap.dedent("""\
            version: "1"
            streams: []
        """)
        p = tmp_path / "empty.yaml"
        p.write_text(content, encoding="utf-8")
        reg = load_taxonomy_registry(p)
        assert len(reg.streams) == 0
        assert len(reg.canonical_label_map) == 0

    def test_optional_fields_have_defaults(self, tmp_path: pathlib.Path):
        content = textwrap.dedent("""\
            version: "1"
            streams:
              - id: 99
                canonical_name: "Minimal Stream"
        """)
        p = tmp_path / "minimal.yaml"
        p.write_text(content, encoding="utf-8")
        reg = load_taxonomy_registry(p)
        s = reg.streams[0]
        assert s.evaluation_name == "Minimal Stream"
        assert s.aliases == []
        assert s.family == ""
        assert s.broad is False
        assert s.overlaps_with == []
        assert s.preferred_over == []
        assert s.suppress_if_preferred is False
