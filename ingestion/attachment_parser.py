"""
Attachment parser (V6).

Splits card text into structured sections that resemble attachments:
exhibits, appendices, tables, scopes, budget sheets, and headed sections.

This is a text-level parser — it does not process binary files. It works
on the already-extracted cleaned card text, identifying structural markers
that indicate the card contains embedded attachment-like content.

For binary attachment parsing (PDF, XLSX, DOCX), callers should pre-extract
text and pass it through pipeline state as _attachment_contents list.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Section-type vocabulary used downstream for scoring decisions
SECTION_TYPE_EXHIBIT = "exhibit"
SECTION_TYPE_APPENDIX = "appendix"
SECTION_TYPE_TABLE = "table"
SECTION_TYPE_BUDGET = "budget"
SECTION_TYPE_SCOPE = "scope"
SECTION_TYPE_ROADMAP = "roadmap"
SECTION_TYPE_REQUIREMENTS = "requirements"
SECTION_TYPE_HEADING = "heading"
SECTION_TYPE_BODY = "body"

# Regex patterns for section detection
_EXHIBIT_RE = re.compile(
    r"(?i)^(exhibit\s+[a-z0-9]+[:\-–—]?\s*)(.{0,120})",
    re.MULTILINE,
)
_APPENDIX_RE = re.compile(
    r"(?i)^(appendix\s+[a-z0-9]+[:\-–—]?\s*)(.{0,120})",
    re.MULTILINE,
)
_TABLE_RE = re.compile(
    r"(?i)^(table\s+[0-9]+[:\-–—]?\s*)(.{0,120})",
    re.MULTILINE,
)
_BUDGET_RE = re.compile(
    r"(?i)^(budget\s*[:\-–—]?\s*)(.{0,120})",
    re.MULTILINE,
)
_SCOPE_RE = re.compile(
    r"(?i)^(scope(\s+of\s+work)?[:\-–—]\s*)(.{0,120})",
    re.MULTILINE,
)
_ROADMAP_RE = re.compile(
    r"(?i)^(roadmap[:\-–—]\s*)(.{0,120})",
    re.MULTILINE,
)
_REQUIREMENTS_RE = re.compile(
    r"(?i)^(requirements?[:\-–—]\s*)(.{0,120})",
    re.MULTILINE,
)
_HEADING_RE = re.compile(
    r"(?m)^([A-Z][A-Z0-9 /&,\-]{4,60}):?\s*$",
)

_ALL_SECTION_PATTERNS = [
    (_EXHIBIT_RE, SECTION_TYPE_EXHIBIT),
    (_APPENDIX_RE, SECTION_TYPE_APPENDIX),
    (_TABLE_RE, SECTION_TYPE_TABLE),
    (_BUDGET_RE, SECTION_TYPE_BUDGET),
    (_SCOPE_RE, SECTION_TYPE_SCOPE),
    (_ROADMAP_RE, SECTION_TYPE_ROADMAP),
    (_REQUIREMENTS_RE, SECTION_TYPE_REQUIREMENTS),
]

# Minimum section content length to be useful
_MIN_SECTION_CHARS = 30


@dataclass
class ParsedSection:
    section_id: str
    section_title: str
    section_type: str
    content: str
    start_offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "section_type": self.section_type,
            "content": self.content[:500],  # cap for state transport
            "start_offset": self.start_offset,
        }


@dataclass
class ParsedAttachment:
    attachment_id: str
    filename: str
    sections: List[ParsedSection] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attachment_id": self.attachment_id,
            "filename": self.filename,
            "section_count": len(self.sections),
            "sections": [s.to_dict() for s in self.sections],
        }


class AttachmentParser:
    """
    Extracts structured sections from card text.

    Handles two input modes:
    1. Card text only — scans cleaned_text for structural markers
    2. Explicit attachment content — processes each entry in _attachment_contents
       (list of {"filename": str, "text": str} dicts)
    """

    def parse_card_text(self, cleaned_text: str) -> Optional[ParsedAttachment]:
        """
        Extract attachment-like sections from a card's cleaned text.

        Returns a ParsedAttachment if any structural sections are found,
        or None if the text has no attachment-like structure.
        """
        if not cleaned_text or not cleaned_text.strip():
            return None

        sections = self._extract_sections(cleaned_text)
        if not sections:
            return None

        return ParsedAttachment(
            attachment_id=str(uuid.uuid4()),
            filename="card_body",
            sections=sections,
            raw_text=cleaned_text[:2000],
        )

    def parse_attachment_content(
        self, filename: str, text: str
    ) -> ParsedAttachment:
        """
        Parse an explicit attachment text document into sections.
        """
        sections = self._extract_sections(text)
        return ParsedAttachment(
            attachment_id=str(uuid.uuid4()),
            filename=filename,
            sections=sections,
            raw_text=text[:2000],
        )

    def _extract_sections(self, text: str) -> List[ParsedSection]:
        """Find all structural section markers in text."""
        # Collect all marker positions
        markers: List[tuple] = []  # (start, end_of_title, section_type, title)

        for pattern, section_type in _ALL_SECTION_PATTERNS:
            for m in pattern.finditer(text):
                title_text = m.group(0).strip()
                markers.append((m.start(), m.end(), section_type, title_text))

        # Also look for ALL-CAPS headings (lower priority)
        for m in _HEADING_RE.finditer(text):
            heading = m.group(1).strip()
            # Skip if already captured by a higher-priority pattern
            overlap = any(abs(m.start() - pos) < 20 for pos, _, _, _ in markers)
            if not overlap:
                markers.append((m.start(), m.end(), SECTION_TYPE_HEADING, heading))

        if not markers:
            return []

        # Sort by position
        markers.sort(key=lambda x: x[0])

        # Extract section content (text between markers)
        sections: List[ParsedSection] = []
        for i, (start, title_end, stype, title) in enumerate(markers):
            # Content runs from end of title to start of next section
            next_start = markers[i + 1][0] if i + 1 < len(markers) else len(text)
            content = text[title_end:next_start].strip()

            if len(content) < _MIN_SECTION_CHARS:
                continue

            sections.append(ParsedSection(
                section_id=str(uuid.uuid4()),
                section_title=title[:80],
                section_type=stype,
                content=content,
                start_offset=start,
            ))

        return sections
