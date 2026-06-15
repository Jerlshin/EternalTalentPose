"""JD latent-family extractor (``features/latents.py``).

Owner layer: features. Allowed imports: ``domain``, ``features.view``, stdlib.

Implements Part 2: the twelve ``jd.*`` latent concepts, each modeled as a
``value + confidence`` built from already-computed upstream cells (competency,
career, product-vs-services, behavioral). Because every latent is a reduction
over other feature cells, this extractor runs last in the extraction DAG and
takes the assembled ``{feature_id: FeatureCell}`` map as input.

Two spec rules are enforced:

* **Anti-stuffing core.** ``jd.keyword_only = clamp(mean(claimed) −
  mean(trust·in_career·semantic))`` — large when a profile lists AI skills with
  no corroboration; it is the negative latent that subtracts from the positives.
* **Sparse profiles are not confidently labeled.** A latent with little
  supporting evidence regresses toward a neutral prior (0.0) weighted by its
  evidence coverage, and emits a coverage-driven confidence the Fit/Risk engines
  read alongside the value.

Evidence is the set of constituent feature ids (``EvidenceKind.DERIVED``), so
the ``feature → feature → raw`` lineage chain stays walkable end to end.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from redstack.domain.enums import EvidenceKind
from redstack.domain.provenance import EvidenceRef
from redstack.features.view import (
    CellEmission,
    FeatureCell,
    cell,
    clamp_unit,
    make_evidence,
    mean_of,
)

_DERIVED = EvidenceKind.DERIVED
_NEUTRAL_PRIOR: Final[float] = 0.0

_COMPETENCY_GROUPS: Final[tuple[str, ...]] = (
    "retr",
    "rank",
    "recsys",
    "ir",
    "nlp",
    "llm",
    "mle",
    "mlops",
    "eval",
)


def _gather(
    upstream: Mapping[str, FeatureCell], specs: tuple[tuple[str, bool], ...]
) -> tuple[list[float], list[EvidenceRef], int]:
    """Collect contributions for present constituents.

    ``specs`` are ``(feature_id, invert)`` pairs; ``invert`` flips the value to
    ``1 - value`` (used for "absence of X" negative evidence). Returns the list
    of contributions, the evidence refs, and the count of present constituents.
    """
    contributions: list[float] = []
    evidence: list[EvidenceRef] = []
    for feature_id, invert in specs:
        found = upstream.get(feature_id)
        if found is None:
            continue
        contribution = (1.0 - found.value) if invert else found.value
        contributions.append(clamp_unit(contribution))
        evidence.append(make_evidence(_DERIVED, feature_id, found.value))
    return contributions, evidence, len(specs)


def _latent(
    feature_id: str,
    upstream: Mapping[str, FeatureCell],
    specs: tuple[tuple[str, bool], ...],
) -> tuple[str, FeatureCell]:
    """Build one latent: mean of present contributions, regressed toward prior."""
    contributions, evidence, expected = _gather(upstream, specs)
    present = len(contributions)
    coverage = present / float(expected) if expected else 0.0
    raw = mean_of(tuple(contributions))
    # Regress toward the neutral prior by (1 - coverage): sparse ⇒ not labeled.
    value = clamp_unit(coverage * raw + (1.0 - coverage) * _NEUTRAL_PRIOR)
    confidence = clamp_unit(0.3 + 0.7 * coverage)
    if not evidence:
        evidence = [make_evidence(_DERIVED, feature_id, value)]
    return feature_id, cell(value, confidence, tuple(evidence))


def _keyword_only(upstream: Mapping[str, FeatureCell]) -> tuple[str, FeatureCell]:
    """``mean(claimed) − mean(trust·in_career·semantic)`` over the nine groups."""
    claimed_values: list[float] = []
    corroboration_values: list[float] = []
    evidence: list[EvidenceRef] = []
    for group in _COMPETENCY_GROUPS:
        claimed = upstream.get(f"{group}.claimed")
        trust = upstream.get(f"{group}.trust")
        in_career = upstream.get(f"{group}.in_career")
        semantic = upstream.get(f"{group}.semantic")
        if claimed is None or trust is None or in_career is None or semantic is None:
            continue
        claimed_values.append(claimed.value)
        corroboration_values.append(
            clamp_unit(trust.value * in_career.value * semantic.value)
        )
        evidence.append(make_evidence(_DERIVED, f"{group}.claimed", claimed.value))

    if not claimed_values:
        return "jd.keyword_only", cell(
            0.0, 0.2, (make_evidence(_DERIVED, "jd.keyword_only", 0.0),)
        )

    value = clamp_unit(mean_of(tuple(claimed_values)) - mean_of(tuple(corroboration_values)))
    coverage = len(claimed_values) / float(len(_COMPETENCY_GROUPS))
    confidence = clamp_unit(0.3 + 0.7 * coverage)
    return "jd.keyword_only", cell(value, confidence, tuple(evidence))


# --------------------------------------------------------------------------- #
# Latent specifications (feature_id → constituent (id, invert) tuples).       #
# --------------------------------------------------------------------------- #
_POSITIVE_SPECS: Final[tuple[tuple[str, tuple[tuple[str, bool], ...]], ...]] = (
    (
        "jd.retrieval_ranking",
        (("retr.competency", False), ("rank.competency", False), ("ir.competency", False)),
    ),
    (
        "jd.production_ml",
        (("mle.competency", False), ("mlops.competency", False), ("pvs.product_density", False)),
    ),
    (
        "jd.product_company",
        (("pvs.product_density", False), ("pvs.product_recent", False)),
    ),
    ("jd.hybrid_retrieval", (("ir.competency", False),)),
    ("jd.eval_framework", (("eval.competency", False),)),
    (
        "jd.shipping_mentality",
        (("startup.shipping_signal", False), ("found.ownership", False)),
    ),
)

_NEGATIVE_SPECS: Final[tuple[tuple[str, tuple[tuple[str, bool], ...]], ...]] = (
    (
        "jd.consulting_only",
        (("pvs.consulting_density", False), ("career.consulting_density", False)),
    ),
    (
        "jd.title_chaser",
        (("career.title_inflation", False), ("career.stability", True)),
    ),
    (
        "jd.pure_researcher",
        (("career.research_only", False), ("career.production_exposure", True)),
    ),
    (
        "jd.framework_enthusiast",
        (("llm.claimed", False), ("llm.trust", True)),
    ),
    (
        "jd.inactive",
        (("avail.available", True), ("resp.reliable", True), ("bhv.availability", True)),
    ),
)


def extract(upstream: Mapping[str, FeatureCell]) -> CellEmission:
    """Emit the twelve ``jd.*`` latent cells from the assembled upstream cells.

    Pure reduction over the ``{feature_id: FeatureCell}`` map produced by the
    competency / career / behavioral extractors. Constituents absent from the
    map lower a latent's coverage (and thus its confidence and magnitude) rather
    than raising an error, so the latent layer composes deterministically.
    """
    rows: list[tuple[str, FeatureCell]] = []
    for feature_id, specs in _POSITIVE_SPECS:
        rows.append(_latent(feature_id, upstream, specs))
    rows.append(_keyword_only(upstream))
    for feature_id, specs in _NEGATIVE_SPECS:
        rows.append(_latent(feature_id, upstream, specs))
    return tuple(rows)


__all__ = ("extract",)