

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

from redstack.config.schema import (
    AnchorPolarity,
    EligibilityRulesConfig,
    JdAnchorsConfig,
)
from redstack.domain.enums import EligibilityCode
from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.registry import (
    OFFLINE_ARTIFACT_REGISTRY,
    OfflineArtifactRegistry,
)
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "JdConceptStage",
    "jd_concept_stage",
)

#: The fixed positive/negative jd.* latent vocabulary (Feature Layer Part 2).
#: Anchor ids are validated to be drawn from this closed set so the authored
#: anchors line up with the latent composition the feature layer computes.
_POSITIVE_LATENTS: Final[frozenset[str]] = frozenset(
    {
        "jd.retrieval_ranking",
        "jd.production_ml",
        "jd.product_company",
        "jd.shipping_mentality",
        "jd.eval_framework",
        "jd.hybrid_retrieval",
    }
)
_NEGATIVE_LATENTS: Final[frozenset[str]] = frozenset(
    {
        "jd.keyword_only",
        "jd.consulting_only",
        "jd.title_chaser",
        "jd.pure_researcher",
        "jd.framework_enthusiast",
        "jd.inactive",
    }
)


class JdConceptStage(OfflineStage):
    """O6 — author jd.* anchors + package the eligibility gate rules."""

    stage_id = "O6"
    stage_version = "1.0"

    def __init__(
        self,
        anchors: JdAnchorsConfig,
        eligibility: EligibilityRulesConfig,
        registry: OfflineArtifactRegistry = OFFLINE_ARTIFACT_REGISTRY,
    ) -> None:
        """Construct with the injected JD authoring seeds.

        Args:
            anchors: ``anchors/jd_anchors.yaml`` parsed — positive/negative intents.
            eligibility: ``gates/eligibility_rules.yaml`` parsed — hard/soft rules.
            registry: the artifact catalog (defaults to the frozen catalog).
        """
        super().__init__(registry)
        self._anchors = anchors
        self._eligibility = eligibility

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        concept_dictionary = self._load_concepts(ctx)
        anchors_payload = self._author_anchors(concept_dictionary)
        gates_payload = self._package_eligibility()

        art_anchors = self.emit_json(ctx, "jd_concepts", anchors_payload)
        art_gates = self.emit_yaml(ctx, "eligibility_rules", gates_payload)

        anchors_list = anchors_payload["anchors"]
        assert isinstance(anchors_list, list)
        metrics: dict[str, object] = {
            "anchor_count": len(anchors_list),
            "positive_anchors": sum(
                1 for a in anchors_list
                if isinstance(a, Mapping) and a.get("polarity") == "positive"
            ),
            "negative_anchors": sum(
                1 for a in anchors_list
                if isinstance(a, Mapping) and a.get("polarity") == "negative"
            ),
            "hard_blocks": len(self._eligibility.hard_blocks),
            "soft_penalties": len(self._eligibility.soft_penalties),
        }
        return StageResult(artifacts=(art_anchors, art_gates), metrics=metrics)

    # ------------------------------------------------------------------ #
    # Concept dictionary load (O5 output)                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_concepts(ctx: OfflinePipelineContext) -> Mapping[str, object]:
        """Load O5's expanded concept dictionary for anchor vocab enrichment.

        A missing/unreadable dictionary is non-fatal for anchor *authoring* (the
        anchor text is human-authored); it only suppresses the optional vocab
        enrichment. Returns an empty mapping if absent.
        """
        try:
            raw = ctx.artifact_store.load_text("concepts")
        except Exception:  # noqa: BLE001 — enrichment is best-effort, not a gate.
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            concepts = parsed.get("concepts")
            if isinstance(concepts, Mapping):
                return concepts
        return {}

    # ------------------------------------------------------------------ #
    # Anchor authoring                                                   #
    # ------------------------------------------------------------------ #
    def _author_anchors(
        self, concept_dictionary: Mapping[str, object]
    ) -> dict[str, object]:
        """Author the polarity-tagged anchor set, enriched with concept vocab.

        Each authored ``AnchorIntent`` becomes an anchor record carrying its id,
        polarity (by value), the human anchor text, and — when the anchor id maps
        to a known concept — that concept's discovered vocab (so the embedded
        anchor text is meaning-rich, the dense anti-stuffing counterpart).

        Validates each anchor id against the fixed ``jd.*`` latent vocabulary and
        confirms its polarity matches the latent's sign.

        Raises:
            ArtifactContractError: an anchor id is outside the known latent set,
                or its polarity contradicts the latent's sign.
        """
        records: list[dict[str, object]] = []
        for intent in self._anchors.anchors:
            self._validate_anchor_id(intent.id, intent.polarity)
            record: dict[str, object] = {
                "id": intent.id,
                "polarity": intent.polarity.value,
                "text": intent.text,
            }
            vocab = self._concept_vocab_for(intent.id, concept_dictionary)
            if vocab:
                record["vocab"] = vocab
            records.append(record)
        records.sort(key=lambda r: str(r["id"]))
        return {
            "anchors": records,
            "positive_latents": sorted(_POSITIVE_LATENTS),
            "negative_latents": sorted(_NEGATIVE_LATENTS),
        }

    @staticmethod
    def _validate_anchor_id(anchor_id: str, polarity: AnchorPolarity) -> None:
        """Assert the anchor id is a known latent and its polarity matches.

        Raises:
            ArtifactContractError: unknown id, or polarity/sign mismatch.
        """
        is_positive = anchor_id in _POSITIVE_LATENTS
        is_negative = anchor_id in _NEGATIVE_LATENTS
        if not (is_positive or is_negative):
            msg = (
                f"anchor id {anchor_id!r} is not a known jd.* latent "
                f"(positive={sorted(_POSITIVE_LATENTS)}, "
                f"negative={sorted(_NEGATIVE_LATENTS)})"
            )
            raise ArtifactContractError(msg)
        if is_positive and polarity is not AnchorPolarity.POSITIVE:
            msg = f"anchor {anchor_id!r} is a positive latent but tagged {polarity.value!r}"
            raise ArtifactContractError(msg)
        if is_negative and polarity is not AnchorPolarity.NEGATIVE:
            msg = f"anchor {anchor_id!r} is a negative latent but tagged {polarity.value!r}"
            raise ArtifactContractError(msg)

    @staticmethod
    def _concept_vocab_for(
        anchor_id: str, concept_dictionary: Mapping[str, object]
    ) -> list[str]:
        """Return the discovered vocab for the concept this anchor maps to.

        The mapping is by the latent's concept suffix (``jd.retrieval_ranking`` →
        the ``retrieval`` / ``ranking`` concept families when present). Lookup is
        deterministic and best-effort: an unmapped anchor simply carries no vocab.
        """
        suffix = anchor_id.split(".", 1)[1] if "." in anchor_id else anchor_id
        keys = sorted(
            name
            for name in concept_dictionary
            if isinstance(name, str) and name and name in suffix
        )
        vocab: list[str] = []
        for name in keys:
            body = concept_dictionary[name]
            if isinstance(body, Mapping):
                terms = body.get("vocab")
                if isinstance(terms, (list, tuple)):
                    vocab.extend(str(t) for t in terms if isinstance(t, str))
        return sorted(set(vocab))

    # ------------------------------------------------------------------ #
    # Eligibility gate packaging                                         #
    # ------------------------------------------------------------------ #
    def _package_eligibility(self) -> dict[str, object]:
        """Package the authored eligibility rules, cross-checking every code.

        Hard blocks and soft penalties are emitted in sorted-by-code order; each
        code is validated against the domain ``EligibilityCode`` vocabulary at
        this artifact boundary (Repo Layout §5) so an unknown or misspelled code
        fails the build rather than silently disabling a gate online.

        Raises:
            ArtifactContractError: a rule code is not a valid ``EligibilityCode``.
        """
        valid = {code.value for code in EligibilityCode}
        hard: list[dict[str, object]] = []
        for rule in self._eligibility.hard_blocks:
            self._check_code(rule.code, valid)
            hard.append({"code": rule.code, "description": rule.description})
        soft: list[dict[str, object]] = []
        for rule in self._eligibility.soft_penalties:
            self._check_code(rule.code, valid)
            soft.append(
                {
                    "code": rule.code,
                    "description": rule.description,
                    "penalty": rule.penalty,
                }
            )
        hard.sort(key=lambda r: str(r["code"]))
        soft.sort(key=lambda r: str(r["code"]))
        return {"hard_blocks": hard, "soft_penalties": soft}

    @staticmethod
    def _check_code(code: str, valid: set[str]) -> None:
        """Assert ``code`` is a member of the domain ``EligibilityCode`` set."""
        if code not in valid:
            msg = (
                f"eligibility code {code!r} is not a valid EligibilityCode; "
                f"known: {sorted(valid)}"
            )
            raise ArtifactContractError(msg)


def jd_concept_stage(
    anchors: JdAnchorsConfig, eligibility: EligibilityRulesConfig
) -> JdConceptStage:
    """Factory: construct the O6 JD-concept stage bound to the injected seeds."""
    return JdConceptStage(anchors, eligibility)