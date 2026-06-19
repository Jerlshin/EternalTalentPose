
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from redstack.domain.enums import EvidenceKind, ReasoningPolarity
from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "ReasoningTemplateStage",
    "reasoning_template_stage",
    "DEFAULT_TEMPLATE_CATALOG_PATH",
)

#: The committed default catalog shipped beside this module.
DEFAULT_TEMPLATE_CATALOG_PATH: Final[Path] = (
    Path(__file__).with_name("reasoning_templates.json")
)

#: Valid rank bands a template may target (Domain §J ``CandidateReasoning``).
_VALID_RANK_BANDS: Final[frozenset[str]] = frozenset({"top", "mid", "tail"})


class ReasoningTemplateStage(OfflineStage):
    """O16 — assemble evidence-slot reasoning templates (LLM-free)."""

    stage_id = "O16"
    stage_version = "1.0"

    def __init__(
        self,
        catalog_path: Path = DEFAULT_TEMPLATE_CATALOG_PATH,
        *,
        registry: object | None = None,
    ) -> None:
        """Construct with the committed default catalog path.

        Args:
            catalog_path: the committed evidence-slot template seed (defaults to
                the sibling ``reasoning_templates.json``).
            registry: optional artifact registry override (kept generic to avoid
                a hard import here; the base default is used when ``None``).
        """
        if registry is None:
            super().__init__()
        else:
            from redstack.pipelines.offline.registry import OfflineArtifactRegistry

            assert isinstance(registry, OfflineArtifactRegistry)
            super().__init__(registry)
        self._catalog_path = catalog_path

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        catalog = self._load_default_catalog()
        cited_features = self._gold_cited_features(ctx)
        importance = self._feature_importance(ctx)

        templates = self._refine_templates(catalog, cited_features, importance)
        if not templates:
            msg = "reasoning template construction produced no templates"
            raise ArtifactContractError(msg)

        payload: dict[str, object] = {
            "schema_version": "reason-v1.0",
            "no_online_llm": True,
            "rank_bands": sorted(_VALID_RANK_BANDS),
            "templates": templates,
        }
        artifact = self.emit_json(ctx, "reasoning_templates", payload)
        metrics: dict[str, object] = {
            "template_count": len(templates),
            "gold_cited_features": len(cited_features),
            "importance_features": len(importance),
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    # ------------------------------------------------------------------ #
    # Catalog + input loading                                            #
    # ------------------------------------------------------------------ #
    def _load_default_catalog(self) -> list[Mapping[str, object]]:
        """Load + structurally validate the committed default template catalog.

        Raises:
            ArtifactContractError: the catalog is missing/unreadable/malformed, or
                any template/slot violates the evidence-kind / polarity / band
                contract (the Stage-4 guarantee, enforced here).
        """
        try:
            raw = self._catalog_path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"cannot read reasoning template catalog at {self._catalog_path}: {exc}"
            raise ArtifactContractError(msg) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"reasoning template catalog is not valid JSON: {exc}"
            raise ArtifactContractError(msg) from exc
        if not isinstance(parsed, Mapping):
            raise ArtifactContractError("reasoning template catalog must be an object")
        templates = parsed.get("templates")
        if not isinstance(templates, (list, tuple)) or not templates:
            raise ArtifactContractError("catalog must declare a non-empty 'templates'")
        validated: list[Mapping[str, object]] = []
        for template in templates:
            if not isinstance(template, Mapping):
                raise ArtifactContractError("each template must be an object")
            self._validate_template(template)
            validated.append(template)
        return validated

    @staticmethod
    def _validate_template(template: Mapping[str, object]) -> None:
        """Assert one template's polarity, rank bands, and evidence slots are valid.

        Raises:
            ArtifactContractError: unknown polarity, empty/unknown rank band, or a
                slot missing / naming an unknown ``EvidenceKind``.
        """
        tid = template.get("id")
        if not isinstance(tid, str) or not tid:
            raise ArtifactContractError("each template must carry a non-empty id")
        polarity = template.get("polarity")
        valid_polarities = {p.value for p in ReasoningPolarity}
        if polarity not in valid_polarities:
            msg = f"template {tid!r} has invalid polarity {polarity!r}"
            raise ArtifactContractError(msg)
        bands = template.get("rank_bands")
        if not isinstance(bands, (list, tuple)) or not bands:
            raise ArtifactContractError(f"template {tid!r} must declare rank_bands")
        for band in bands:
            if band not in _VALID_RANK_BANDS:
                msg = f"template {tid!r} has invalid rank band {band!r}"
                raise ArtifactContractError(msg)
        slots = template.get("slots")
        if not isinstance(slots, (list, tuple)) or not slots:
            raise ArtifactContractError(f"template {tid!r} must declare slots")
        valid_kinds = {k.value for k in EvidenceKind}
        for slot in slots:
            if not isinstance(slot, Mapping):
                raise ArtifactContractError(f"template {tid!r} has a non-object slot")
            kind = slot.get("evidence_kind")
            if kind not in valid_kinds:
                msg = (
                    f"template {tid!r} slot {slot.get('name')!r} has invalid "
                    f"evidence_kind {kind!r}; known: {sorted(valid_kinds)}"
                )
                raise ArtifactContractError(msg)

    @staticmethod
    def _gold_cited_features(ctx: OfflinePipelineContext) -> frozenset[str]:
        """Mine which features the gold human reasonings cited (best-effort).

        Reads O8's ``gold_labels.json`` if present and extracts any
        ``cited_features`` recorded per label; the union is used to prioritize
        templates whose source feature reviewers actually relied on. Absent gold
        (early build) yields an empty set — templates still emit from the catalog.
        """
        try:
            raw = ctx.artifact_store.load_text("gold_labels")
        except Exception:  # noqa: BLE001 — gold may not exist yet; non-fatal.
            return frozenset()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return frozenset()
        cited: set[str] = set()
        if isinstance(parsed, Mapping):
            labels = parsed.get("labels")
            if isinstance(labels, Mapping):
                for record in labels.values():
                    if isinstance(record, Mapping):
                        feats = record.get("cited_features")
                        if isinstance(feats, (list, tuple)):
                            cited.update(str(f) for f in feats if isinstance(f, str))
        return frozenset(cited)

    @staticmethod
    def _feature_importance(ctx: OfflinePipelineContext) -> Mapping[str, float]:
        """Load O10 feature importances (best-effort) for template ordering.

        Absent importance (early build) yields an empty mapping; ordering then
        falls back to template id (still deterministic).
        """
        try:
            raw = ctx.artifact_store.load_text("feature_importance")
        except Exception:  # noqa: BLE001 — importance may not exist yet; non-fatal.
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            importances = parsed.get("importances")
            if isinstance(importances, Mapping):
                return {
                    str(k): float(v)
                    for k, v in importances.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
        return {}

    # ------------------------------------------------------------------ #
    # Template refinement                                                #
    # ------------------------------------------------------------------ #
    def _refine_templates(
        self,
        catalog: Sequence[Mapping[str, object]],
        cited_features: frozenset[str],
        importance: Mapping[str, float],
    ) -> list[dict[str, object]]:
        """Annotate catalog templates with gold support + importance, sort stably.

        Each template gains ``gold_supported`` (any slot's source feature was
        cited by a human reviewer) and ``priority`` (max source-feature importance
        across its slots). Templates are emitted sorted by ``id`` so the artifact
        bytes are deterministic; ``priority`` is data the online engine reads to
        order clause selection, not a sort key here.
        """
        refined: list[dict[str, object]] = []
        for template in catalog:
            slots = template.get("slots")
            assert isinstance(slots, (list, tuple))
            source_features = [
                str(slot["source_feature"])
                for slot in slots
                if isinstance(slot, Mapping) and "source_feature" in slot
            ]
            gold_supported = any(f in cited_features for f in source_features)
            priority = max(
                (importance.get(f, 0.0) for f in source_features), default=0.0
            )
            bands = template.get("rank_bands")
            rank_bands = [str(b) for b in bands] if isinstance(bands, (list, tuple)) else []
            record: dict[str, object] = {
                "id": template["id"],
                "polarity": template["polarity"],
                "rank_bands": rank_bands,
                "jd_link": template.get("jd_link"),
                "fragment": template.get("fragment", ""),
                "slots": [dict(slot) for slot in slots if isinstance(slot, Mapping)],
                "gold_supported": gold_supported,
                "priority": round(priority, 6),
            }
            refined.append(record)
        refined.sort(key=lambda r: str(r["id"]))
        return refined


def reasoning_template_stage(
    catalog_path: Path = DEFAULT_TEMPLATE_CATALOG_PATH,
) -> ReasoningTemplateStage:
    """Factory: construct the O16 reasoning-template stage."""
    return ReasoningTemplateStage(catalog_path)