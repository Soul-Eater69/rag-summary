"""
Adapter layer (V5 architecture).

Thin Protocol interfaces for the three external service dependencies:
- LLMService: text generation (wraps src.services.generation_service)
- EmbeddingService: vector embeddings (wraps src.clients.embedding)
- KGRetrievalService: knowledge-graph candidate retrieval (wraps
  src.pipelines.value_stream.retrieval_pipeline)

Usage
-----
Modules should accept an optional adapter parameter::

    def generate_new_card_summary(
        ppt_text: str,
        *,
        llm: LLMService | None = None,
    ) -> SummaryDoc:
        svc = llm or get_default_llm()
        ...

Tests can inject stub implementations without touching src.*::

    class StubLLM:
        def generate(self, query, *, context="", system_prompt=""):
            return SimpleNamespace(content='{"short_summary": "stub"}')

    summary = generate_new_card_summary(text, llm=StubLLM())

The factory functions (get_default_*) lazy-import from src.* so the
adapters module itself is importable even when src.* is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Protocol, Type, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Protocols (structural interfaces)
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMResponse(Protocol):
    """Minimal contract for an LLM response object."""
    content: str


@runtime_checkable
class LLMService(Protocol):
    """Text generation service interface."""

    def generate(
        self,
        query: str,
        *,
        context: str = "",
        system_prompt: str = "",
    ) -> LLMResponse:
        ...


@runtime_checkable
class StructuredLLMService(LLMService, Protocol):
    """
    LLM service that supports provider-native structured output binding.

    Providers that support this (e.g. OpenAI, Anthropic tool-use) can
    implement generate_structured to return a validated Pydantic model
    instance directly, without round-tripping through JSON text + parse.
    """

    def generate_structured(
        self,
        query: str,
        output_schema: type,
        *,
        context: str = "",
        system_prompt: str = "",
    ) -> Any:
        ...


@runtime_checkable
class EmbeddingService(Protocol):
    """Vector embedding service interface."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


@runtime_checkable
class KGRetrievalService(Protocol):
    """Knowledge-graph candidate retrieval interface."""

    def retrieve_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 20,
        allowed_names: Optional[List[str]] = None,
    ) -> List[dict]:
        ...


@runtime_checkable
class ThemeRetrievalService(Protocol):
    """
    Theme-cluster candidate retrieval interface.

    Implementations should return candidates shaped as:
    {"entity_name": str, "score": float, "theme_cluster": str, "description": str}

    The default implementation (_NoopThemeService) returns an empty list.
    Wire a real implementation to activate the theme evidence source.
    """

    def retrieve_theme_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        allowed_names: Optional[List[str]] = None,
    ) -> List[dict]:
        ...


@runtime_checkable
class SummaryIndexService(Protocol):
    """
    Protocol for searching the FAISS summary index for analog tickets.

    Abstracts FaissIndexer so nodes can be tested without the real index.
    """

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        allowed_vs_names: Optional[List[str]] = None,
    ) -> List[dict]:
        ...


@runtime_checkable
class RawEvidenceService(Protocol):
    """
    Protocol for fetching raw evidence chunks for a specific ticket.

    Abstracts the filesystem ticket_chunks/ lookup so nodes can be tested
    without the actual chunk files on disk.
    """

    def get_chunks_for_ticket(
        self,
        ticket_id: str,
        *,
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> List[dict]:
        ...

# ---------------------------------------------------------------------------
# Default factory functions (lazy import from src.*)
# ---------------------------------------------------------------------------

def get_default_llm() -> LLMService:
    """Return the default LLM service from the internal codebase."""
    from src.services.generation_service import GenerationService  # type: ignore[import]
    return GenerationService()


def get_default_embedding() -> EmbeddingService:
    """Return the default embedding client from the internal codebase."""
    from src.clients.embedding import EmbeddingClient  # type: ignore[import]
    return EmbeddingClient()


class _KGRetrievalAdapter:
    """Thin adapter wrapping the internal KG retrieval pipeline."""

    def retrieve_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 20,
        allowed_names: Optional[List[str]] = None,
    ) -> List[dict]:
        from src.pipelines.value_stream.retrieval_pipeline import (  # type: ignore[import]
            retrieve_kg_candidates as _retrieve,
        )
        return _retrieve(query_text, top_k=top_k, allowed_names=allowed_names)


def get_default_kg() -> KGRetrievalService:
    """Return the default KG retrieval adapter."""
    return _KGRetrievalAdapter()


class _NoopThemeService:
    """
    Default theme retrieval service - inactive until a real implementation is wired.

    Returns an empty list so the theme source slot stays at 0 without errors.
    Replace with a real ThemeRetrievalService implementation to activate.

    Example (in run_prediction_graph / pipeline config):
        from my_infra import ThemeSearchClient
        pipeline_result = run_summary_rag_pipeline(
            ppt_text,
            theme_svc=ThemeSearchClient(),
        )
    """

    def retrieve_theme_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        allowed_names: Optional[List[str]] = None,
    ) -> List[dict]:
        return []


def get_default_theme() -> ThemeRetrievalService:
    """
    Return the default theme service, configured by config/theme_source.yaml.

    Resolution order (when no explicit _theme_svc is injected):
    1. If backend=faiss (or auto) and FAISS index exists -> FaissThemeRetrievalService
    2. If backend=keyword (or auto, no index) -> KeywordThemeService (always active)
    3. If backend=noop or enabled=false -> _NoopThemeService

    This means theme evidence is never silently zero on a fresh repo:
    KeywordThemeService activates immediately using capability map cues.
    """
    import os
    import pathlib

    config_path = pathlib.Path(__file__).resolve().parent.parent / "config" / "theme_source.yaml"
    config: dict = {}
    if config_path.exists():
        try:
            import yaml  # type: ignore
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("[get_default_theme] Failed to load theme_source.yaml: %s", exc)

    if not config.get("enabled", True) or config.get("backend") == "noop":
        return _NoopThemeService()  # type: ignore[return-value]

    backend = config.get("backend", "auto")
    theme_index_dir = config.get("theme_index_dir", "config/theme_index")

    if backend in ("auto", "faiss"):
        index_file = os.path.join(theme_index_dir, "theme_index.faiss")
        if os.path.exists(index_file):
            try:
                from .theme_retrieval_service import FaissThemeRetrievalService
                embedding_svc = get_default_embedding()
                return FaissThemeRetrievalService(  # type: ignore[return-value]
                    theme_index_dir=theme_index_dir,
                    embedding_svc=embedding_svc,
                    min_vs_support_fraction=config.get("min_vs_support_fraction", 0.30),
                )
            except Exception as exc:
                logger.warning("[get_default_theme] FAISS theme load failed: %s", exc)

    if backend in ("auto", "keyword"):
        try:
            from .keyword_theme_service import KeywordThemeService
            return KeywordThemeService()  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("[get_default_theme] KeywordThemeService load failed: %s", exc)

    return _NoopThemeService()  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Structured output helper
# ---------------------------------------------------------------------------

def structured_generate(
    svc: LLMService,
    query: str,
    output_schema: Type[T],
    *,
    context: str = "",
    system_prompt: str = "",
    debug_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> T:
    """
    Generate with structured output binding.

    Primary path: call svc.generate_structured() if the service implements it
    (i.e. supports provider-native structured output such as OpenAI json_schema
    or Anthropic tool-use).

    Fallback: call svc.generate(), extract JSON, validate with Pydantic.
    A compact schema hint is appended to the query so the model knows what
    fields to produce even when the prompt itself omits the full skeleton.

    Args:
        svc:             LLMService (or StructuredLLMService) instance.
        query:           Rendered user-prompt text.
        output_schema:   Pydantic BaseModel subclass to validate against.
        context:         Optional context string passed to the LLM.
        system_prompt:   System prompt text.

    Returns:
        A validated instance of output_schema.

    Raises:
        ValueError: if fallback parsing fails after exhausting retries.
    """
    # --- Primary: native structured output ---
    if hasattr(svc, "generate_structured"):
        try:
            result = svc.generate_structured(  # type: ignore[attr-defined]
                query,
                output_schema,
                context=context,
                system_prompt=system_prompt,
            )
            if debug_callback is not None:
                debug_callback(
                    {
                        "path": "native_structured",
                        "query": query,
                        "context": context,
                        "system_prompt": system_prompt,
                        "result_type": type(result).__name__,
                        "result": result.model_dump() if hasattr(result, "model_dump") else result,
                    }
                )
            if isinstance(result, output_schema):
                return result
        except Exception as exc:
            logger.warning("[structured_generate] Native path failed, falling back: %s", exc)

    # --- Fallback: text generation + JSON parse + Pydantic validation ---
    schema_hint = _build_schema_hint(output_schema)
    augmented_query = f"{query}\n\n{schema_hint}" if schema_hint else query

    reply = svc.generate(augmented_query, context=context, system_prompt=system_prompt)
    raw_text = reply.content if hasattr(reply, "content") else str(reply)
    parsed = safe_json_extract(raw_text)

    if debug_callback is not None:
        debug_callback(
            {
                "path": "fallback_generate",
                "query": query,
                "augmented_query": augmented_query,
                "context": context,
                "system_prompt": system_prompt,
                "raw_text": raw_text,
                "parsed": parsed,
            }
        )

    try:
        if isinstance(parsed, list):
            # For list-based schemas, try wrapping in a list field
            first_list_field = _find_list_field(output_schema)
            if first_list_field:
                parsed = {first_list_field: parsed}
        return output_schema.model_validate(parsed)
    except Exception as exc:
        logger.error("[structured_generate] Pydantic validation failed: %s | raw=%r", exc, raw_text[:200])
        # Return empty model as last-resort fallback
        try:
            return output_schema()
        except Exception:
            raise ValueError(f"structured_generate: could not produce {output_schema.__name__}") from exc


def _build_schema_hint(schema: type) -> str:
    """
    Build a compact field-list hint appended to prompts when native
    structured output is not available.

    Only generates hints for Pydantic BaseModel subclasses.
    """
    try:
        from pydantic import BaseModel
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return ""
        fields = list(schema.model_fields.keys())
        return f"Return a JSON object with these fields: {{', '.join(fields)}}"
    except Exception:
        return ""


def _find_list_field(schema: type) -> Optional[str]:
    """Return the name of the first list-typed field in a Pydantic model."""
    try:
        import inspect
        from pydantic import BaseModel
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return None
        for name, field in schema.model_fields.items():
            annotation = field.annotation
            origin = getattr(annotation, "__origin__", None)
            if origin is list:
                return name
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Utility: safe JSON extraction (wraps core.prompts)
# ---------------------------------------------------------------------------

def safe_json_extract(text: str) -> Any:
    """
    Parse JSON from an LLM response string.

    Tries core.prompts.safe_json_extract first; falls back to a simple
    inline implementation so the adapter module is usable without core.*.
    """
    try:
        from core.prompts import safe_json_extract as _core_extract  # type: ignore[import]
        return _core_extract(text)
    except Exception:
        pass

    # Fallback: find the first {...} or [...] block and parse it
    import json
    import re
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def normalize_text(text: str) -> str:
    """
    Lowercase, strip, and normalize text for search/comparison.

    Tries core.text.normalize_for_search first; falls back to simple
    lower+strip so the module is usable without core.*.
    """
    try:
        from core.text import normalize_for_search  # type: ignore[import]
        return normalize_for_search(text or "")
    except Exception:
        return (text or "").lower().strip()


def clean_card_text(text: str) -> str:
    """
    Clean/normalize raw PPT / idea card text.

    Tries core.text.clean_ppt_text first; falls back to basic whitespace
    normalization.
    """
    try:
        from core.text import clean_ppt_text  # type: ignore[import]
        return clean_ppt_text(text or "")
    except Exception:
        import re
        return re.sub(r"\s+", " ", (text or "")).strip()
