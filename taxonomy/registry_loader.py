"""
Taxonomy registry loader.

Reads config/taxonomy_registry.yaml and returns a fully-indexed
TaxonomyRegistry instance.

Usage:
    from rag_summary.taxonomy import load_taxonomy_registry

    registry = load_taxonomy_registry()                       # default path
    registry = load_taxonomy_registry("path/to/custom.yaml")  # custom path
"""

from __future__ import annotations

import logging
import pathlib
from typing import Union

import yaml

from rag_summary.models.taxonomy import TaxonomyRegistry, TaxonomyStream

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "config" / "taxonomy_registry.yaml"
)


class TaxonomyRegistryError(Exception):
    """Raised when taxonomy_registry.yaml is malformed or unreadable."""


def load_taxonomy_registry(
    path: Union[str, pathlib.Path, None] = None,
) -> TaxonomyRegistry:
    """
    Load and validate the taxonomy registry from YAML.

    Args:
        path: Path to taxonomy_registry.yaml.  Defaults to
              config/taxonomy_registry.yaml relative to the repo root.

    Returns:
        A fully-indexed TaxonomyRegistry with all lookup maps populated.

    Raises:
        TaxonomyRegistryError: If the file is missing, unparseable, or
            fails schema validation.
    """
    config_path = pathlib.Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise TaxonomyRegistryError(
            f"Taxonomy registry not found: {config_path}. "
            f"Expected a YAML file with 'version' and 'streams' keys."
        )

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise TaxonomyRegistryError(
            f"Failed to parse taxonomy registry YAML at {config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise TaxonomyRegistryError(
            f"Taxonomy registry at {config_path} must be a YAML mapping, "
            f"got {type(raw).__name__}."
        )

    version = raw.get("version", "1")
    raw_streams = raw.get("streams")

    if raw_streams is None:
        raise TaxonomyRegistryError(
            f"Taxonomy registry at {config_path} is missing the required 'streams' key."
        )

    if not isinstance(raw_streams, list):
        raise TaxonomyRegistryError(
            f"'streams' must be a list in {config_path}, got {type(raw_streams).__name__}."
        )

    # Validate each stream entry
    streams = []
    seen_ids = set()
    seen_names = set()

    for i, entry in enumerate(raw_streams):
        if not isinstance(entry, dict):
            raise TaxonomyRegistryError(
                f"Stream entry {i} in {config_path} must be a mapping, "
                f"got {type(entry).__name__}."
            )

        # Required fields
        if "id" not in entry:
            raise TaxonomyRegistryError(
                f"Stream entry {i} in {config_path} is missing required field 'id'."
            )
        if "canonical_name" not in entry:
            raise TaxonomyRegistryError(
                f"Stream entry {i} (id={entry.get('id')}) in {config_path} "
                f"is missing required field 'canonical_name'."
            )

        # Uniqueness checks
        sid = entry["id"]
        sname = entry["canonical_name"]

        if sid in seen_ids:
            raise TaxonomyRegistryError(
                f"Duplicate stream id={sid} in {config_path}."
            )
        if sname in seen_names:
            raise TaxonomyRegistryError(
                f"Duplicate canonical_name='{sname}' in {config_path}."
            )
        seen_ids.add(sid)
        seen_names.add(sname)

        try:
            stream = TaxonomyStream(**entry)
        except Exception as exc:
            raise TaxonomyRegistryError(
                f"Invalid stream entry {i} (id={sid}, name='{sname}') "
                f"in {config_path}: {exc}"
            ) from exc

        streams.append(stream)

    registry = TaxonomyRegistry(version=str(version), streams=streams)
    registry.build_indices()

    # Cross-reference validation: overlaps_with and preferred_over must
    # reference known canonical names.
    all_names = {s.canonical_name for s in streams}
    for stream in streams:
        for ref in stream.overlaps_with:
            if ref not in all_names:
                logger.warning(
                    "taxonomy_registry: stream '%s' references unknown overlap '%s'",
                    stream.canonical_name, ref,
                )
        for ref in stream.preferred_over:
            if ref not in all_names:
                logger.warning(
                    "taxonomy_registry: stream '%s' references unknown preferred_over '%s'",
                    stream.canonical_name, ref,
                )

    logger.info(
        "Loaded taxonomy registry v%s: %d streams, %d aliases, %d families",
        registry.version,
        len(registry.streams),
        len(registry.canonical_label_map),
        len(registry.family_index),
    )
    return registry


def try_load_taxonomy_registry(
    path: Union[str, pathlib.Path, None] = None,
) -> TaxonomyRegistry | None:
    """
    Try to load the taxonomy registry; return None on failure.

    Logs a warning instead of raising. Useful for backward-compatible
    injection where the registry is optional.
    """
    try:
        return load_taxonomy_registry(path)
    except TaxonomyRegistryError as exc:
        logger.warning("taxonomy_registry: %s — taxonomy features disabled", exc)
        return None
    except Exception as exc:
        logger.warning("taxonomy_registry: unexpected error: %s", exc)
        return None
