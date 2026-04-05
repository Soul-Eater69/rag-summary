"""
YAML prompt loader and template renderer.

Loads versioned prompt specs from the prompts/ directory and renders
them with variable substitution.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_PROMPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str, *, version: str = "v1") -> Dict[str, Any]:
    """
    Load a prompt spec by logical name and version.

    Searches prompts/<category>/<name>.<version>.yaml and
    prompts/<name>.<version>.yaml as fallback.

    Returns dict with keys: name, version, system, user, description.
    """
    # Try subdirectory structure first (e.g., prompts/summary/idea_card_summary.v1.yaml)
    candidates = list(_PROMPTS_DIR.rglob(f"{name}.{version}.yaml"))
    if not candidates:
        # Try direct path (e.g., prompts/idea_card_summary.v1.yaml)
        direct = _PROMPTS_DIR / f"{name}.{version}.yaml"
        if direct.exists():
            candidates = [direct]

    if not candidates:
        raise FileNotFoundError(
            f"Prompt '{name}' version '{version}' not found under {_PROMPTS_DIR}"
        )

    prompt_path = candidates[0]
    with prompt_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    return {
        "name": spec.get("name", name),
        "version": spec.get("version", version),
        "description": spec.get("description", ""),
        "system": spec.get("system", ""),
        "user": spec.get("user", ""),
        "_path": str(prompt_path),
    }


def render_prompt(
    prompt_spec: Dict[str, Any],
    variables: Dict[str, Any],
    *,
    role: str = "user",
) -> str:
    """
    Render a prompt template with variable substitution.

    Uses Python str.format_map() — variables in the template are {variable_name}.
    Literal braces that are not variables must be doubled: {{ and }}.

    Args:
        prompt_spec: loaded prompt spec from load_prompt()
        variables: dict of variable values to substitute
        role: "user" or "system" — which template to render

    Returns:
        Rendered string with variables substituted.
    """
    template = prompt_spec.get(role, "")
    if not template:
        return ""

    try:
        return template.format_map(_SafeFormatMap(variables))
    except Exception as exc:
        logger.warning("Prompt render error for '%s' role='%s': %s", prompt_spec.get("name"), role, exc)
        return template


class _SafeFormatMap(dict):
    """Format map that leaves missing keys as-is instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
