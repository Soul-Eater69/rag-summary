"""
Ticket artifact loader: reads raw ticket chunk files from disk and extracts
the text, title, and value-stream labels needed to generate a summary.

This module is intentionally separate from faiss_indexer.py so that the
FAISS layer only handles vector-store operations and nothing about the
on-disk artifact structure.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import List

logger = logging.getLogger(__name__)

def load_ticket_retrieval_text(ticket_dir: pathlib.Path) -> str:
    """
    Return concatenated retrieval text from the richest available chunk file.

    Files are tried in priority order:
    1. `07_chunks.json`          - pre-chunked retrieval chunks (current)
    2. `03_retrieval_views.json` - pre-built retrieval views (legacy)
    3. `02_attachment_text.json` - attachment/PPT text
    4. `01_ticket_data.json`     - raw Jira ticket fields
    """
    for candidate in [
        "07_chunks.json",
        "03_retrieval_views.json",
        "02_attachment_text.json",
        "01_ticket_data.json",
    ]:
        path = ticket_dir / candidate
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = _extract_text_from_artifact(data)
                if text:
                    return text
        except Exception:
            continue
    return ""

def load_ticket_vs_labels(ticket_dir: pathlib.Path) -> List[str]:
    """Load value-stream labels from `08_valuestream_map.json`."""
    vs_map_path = ticket_dir / "08_valuestream_map.json"
    if not vs_map_path.exists():
        return []
    try:
        with open(vs_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("valueStreamNames", []) or []
    except Exception:
        return []

def load_ticket_title(ticket_dir: pathlib.Path, fallback: str) -> str:
    """
    Load the ticket title from `08_valuestream_map.json` or `01_ticket_data.json`.

    Strips a leading ``TICKETID: `` prefix when present.
    """
    for candidate in ["08_valuestream_map.json", "01_ticket_data.json"]:
        path = ticket_dir / candidate
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                title = data.get("title") or data.get("summary") or ""
                if title:
                    return title.split(": ", 1)[1] if ": " in title else title
        except Exception:
            continue
    return fallback

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _extract_text_from_artifact(data: object) -> str:
    """Recursively extract text strings from a parsed JSON artifact."""
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        parts = []
        for val in data.values():
            if isinstance(val, str) and len(val) > 20:
                parts.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        text = (
                            item.get("text")
                            or item.get("content")
                            or item.get("retrieval_text")
                            or ""
                        )
                        if text:
                            parts.append(str(text))
                    elif isinstance(item, str) and len(item) > 20:
                        parts.append(item)
        return "\n\n".join(parts)

    return ""
