

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Final

from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage
from redstack.ports._types import SourceMalformed, SourceOk

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "EMBEDDING_DOC_RECIPE_VERSION",
    "EMBEDDING_DOC_RECIPE",
    "EMBEDDING_FIELD_SEPARATOR",
    "EMBEDDING_SECTION_SEPARATOR",
    "normalize_text",
    "normalize_token",
    "compose_embedding_document",
    "NormalizationStage",
    "normalization_stage",
)

# --------------------------------------------------------------------------- #
# The PINNED embedding-document recipe.                                       #
#                                                                             #
# This ordered tuple of (section_label, source_path) is the single source of #
# truth for how a candidate's free text is concatenated into the document the #
# encoder embeds. ANY change here is a *major* recipe bump (it shifts the     #
# embedding space) and must be reflected in embedding_manifest.json so the    #
# online fallback composes identically. The order is deliberate: identity and #
# headline first (highest signal), then summary, then career descriptions     #
# newest-first, then skills, then education — matching the salience order the  #
# JD reads candidates in. Section labels are embedded verbatim so the encoder  #
# sees a stable, self-describing document.                                     #
# --------------------------------------------------------------------------- #
EMBEDDING_DOC_RECIPE_VERSION: Final[str] = "recipe-v1.0"

#: Separator between fields *within* a section (single space; NFC-stable).
EMBEDDING_FIELD_SEPARATOR: Final[str] = " "
#: Separator between sections (single newline; byte-stable, no trailing space).
EMBEDDING_SECTION_SEPARATOR: Final[str] = "\n"

#: Ordered (label, dotted-source-path) pairs. ``[]`` denotes "iterate the list".
EMBEDDING_DOC_RECIPE: Final[tuple[tuple[str, str], ...]] = (
    ("title", "profile.current_title"),
    ("company", "profile.current_company"),
    ("headline", "profile.headline"),
    ("summary", "profile.summary"),
    ("experience", "career_history[].title|career_history[].description"),
    ("skills", "skills[].name"),
    ("education", "education[].degree|education[].field_of_study"),
)


def normalize_text(value: str) -> str:
    """Canonicalize free text deterministically (NFC, collapse whitespace).

    Applies Unicode NFC normalization, strips, and collapses any run of
    whitespace to a single space. Pure and idempotent: ``normalize_text(x) ==
    normalize_text(normalize_text(x))``. Case is *preserved* for document
    composition (the encoder is case-aware); token canonicalization lowercases
    separately via :func:`normalize_token`.
    """
    nfc = unicodedata.normalize("NFC", value)
    return " ".join(nfc.split())


def normalize_token(value: str) -> str:
    """Canonicalize a skill/company token: NFC, lowercase, whitespace-collapsed.

    The canonical surface form used as a map *value* and as a stable dedup key.
    Idempotent; never empty for a non-blank input (callers skip blanks).
    """
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _resolve_scalar(raw: Mapping[str, object], dotted: str) -> str:
    """Resolve a non-iterating dotted path (e.g. ``profile.headline``) to text.

    Returns the NFC-normalized text, or ``""`` when the path is absent or
    non-string. Never raises — missing free text is empty, not an error (O2
    handles structural validation).
    """
    cursor: object = raw
    for part in dotted.split("."):
        if isinstance(cursor, Mapping) and part in cursor:
            cursor = cursor[part]
        else:
            return ""
    if isinstance(cursor, str):
        return normalize_text(cursor)
    return ""


def _resolve_list_field(raw: Mapping[str, object], spec: str) -> list[str]:
    """Resolve an iterating recipe field (``list[].a|list[].b``) to ordered text.

    ``spec`` is one or more ``<list>[].<attr>`` paths joined by ``|``; all paths
    must reference the same list. For each element, the referenced attributes are
    field-joined; elements are emitted in source order (career history is assumed
    newest-first per the schema, preserving the recipe's salience intent).
    """
    paths = spec.split("|")
    list_key = paths[0].split("[]")[0]
    attrs = [p.split("[].", 1)[1] for p in paths]
    container = raw.get(list_key)
    if not isinstance(container, (list, tuple)):
        return []
    rendered: list[str] = []
    for element in container:
        if not isinstance(element, Mapping):
            continue
        parts: list[str] = []
        for attr in attrs:
            value = element.get(attr)
            if isinstance(value, str) and value.strip():
                parts.append(normalize_text(value))
        if parts:
            rendered.append(EMBEDDING_FIELD_SEPARATOR.join(parts))
    return rendered


def compose_embedding_document(raw: Mapping[str, object]) -> str:
    """Compose the candidate's embedding document per the pinned recipe.

    Walks :data:`EMBEDDING_DOC_RECIPE` in fixed order, rendering each section as
    ``"<label>: <field-joined text>"`` and joining non-empty sections with
    :data:`EMBEDDING_SECTION_SEPARATOR`. The output is byte-deterministic for a
    given record: this exact string is what O13 encodes and what the online ONNX
    fallback must reproduce to land in the same vector space.

    Empty sections are omitted entirely (no dangling labels), keeping the
    document compact while preserving order for the sections that are present.
    """
    sections: list[str] = []
    for label, source in EMBEDDING_DOC_RECIPE:
        if "[]" in source:
            items = _resolve_list_field(raw, source)
            body = EMBEDDING_FIELD_SEPARATOR.join(items)
        else:
            body = _resolve_scalar(raw, source)
        if body:
            sections.append(f"{label}: {body}")
    return EMBEDDING_SECTION_SEPARATOR.join(sections)


class NormalizationStage(OfflineStage):
    """O1 — stream the pool, build canonical maps, pin the embedding recipe."""

    stage_id = "O1"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        token_map: dict[str, str] = {}
        company_map: dict[str, str] = {}
        skill_map: dict[str, str] = {}
        normalized_count = 0
        composed_count = 0

        for record in ctx.candidate_source.stream():
            if isinstance(record, SourceMalformed):
                continue
            if not isinstance(record, SourceOk):
                continue
            normalized_count += 1
            raw = record.raw
            self._accumulate_company(raw, company_map)
            self._accumulate_skills(raw, skill_map, token_map)
            # Compose the document to exercise the recipe deterministically and
            # confirm it renders; the bytes are consumed by O13, not stored here.
            if compose_embedding_document(raw):
                composed_count += 1

        # Canonical maps must be total + non-empty (registry validator). The maps
        # are surface_form -> canonical_form; identity entries are valid canon.
        canonical_maps: dict[str, object] = {
            "recipe_version": EMBEDDING_DOC_RECIPE_VERSION,
            "token_map": dict(sorted(token_map.items())),
            "company_map": dict(sorted(company_map.items())),
            "skill_map": dict(sorted(skill_map.items())),
        }
        artifact = self.emit_json(ctx, "canonical_maps", canonical_maps)
        metrics: dict[str, object] = {
            "normalized_count": normalized_count,
            "composed_documents": composed_count,
            "distinct_skills": len(skill_map),
            "distinct_companies": len(company_map),
            "recipe_version": EMBEDDING_DOC_RECIPE_VERSION,
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    @staticmethod
    def _accumulate_company(
        raw: Mapping[str, object], company_map: dict[str, str]
    ) -> None:
        """Add the current company (and each historical company) to the map."""
        profile = raw.get("profile")
        if isinstance(profile, Mapping):
            current = profile.get("current_company")
            if isinstance(current, str) and current.strip():
                company_map.setdefault(current, normalize_token(current))
        history = raw.get("career_history")
        if isinstance(history, (list, tuple)):
            for position in history:
                if isinstance(position, Mapping):
                    company = position.get("company")
                    if isinstance(company, str) and company.strip():
                        company_map.setdefault(company, normalize_token(company))

    @staticmethod
    def _accumulate_skills(
        raw: Mapping[str, object],
        skill_map: dict[str, str],
        token_map: dict[str, str],
    ) -> None:
        """Add each skill surface form to the skill map and the global token map."""
        skills = raw.get("skills")
        if not isinstance(skills, (list, tuple)):
            return
        for skill in skills:
            if not isinstance(skill, Mapping):
                continue
            name = skill.get("name")
            if isinstance(name, str) and name.strip():
                canon = normalize_token(name)
                skill_map.setdefault(name, canon)
                token_map.setdefault(name, canon)


def normalization_stage() -> NormalizationStage:
    """Factory: construct the O1 normalization stage bound to the frozen registry."""
    return NormalizationStage()