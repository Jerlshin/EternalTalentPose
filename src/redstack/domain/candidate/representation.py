"""``CandidateRepresentation`` — the aggregate root threaded R1->R7.

Owner layer: domain.
Allowed imports: all domain siblings, pydantic.

State model: optional slices + a monotonic ``BuildStage`` discriminant. Slices
attach copy-on-write via ``with_*`` methods that reject regressions and
overwrites; ``require_*`` accessors raise ``RepresentationStageError`` if a
slice is read before population. The set of populated slices must always match
``stage`` exactly (validated at construction), so an illegal partial state is
unrepresentable.
"""

from __future__ import annotations

from typing import Final, final

from pydantic import BaseModel, ConfigDict, model_validator

from redstack.domain.candidate.archetype import ArchetypeAssignment
from redstack.domain.candidate.behavioral import BehavioralProfile
from redstack.domain.candidate.career import CareerProfile
from redstack.domain.candidate.credibility import CredibilityProfile
from redstack.domain.candidate.eligibility import EligibilityReport
from redstack.domain.candidate.identity import Identity
from redstack.domain.candidate.integrity import IntegrityReport
from redstack.domain.candidate.logistics import LogisticsProfile
from redstack.domain.candidate.quality import CandidateQualityVector
from redstack.domain.candidate.semantic import SemanticProfile
from redstack.domain.enums import BuildStage
from redstack.domain.errors import RepresentationStageError
from redstack.domain.ids import CandidateId
from redstack.domain.source import RawCandidate

_FEATURE_SLICES: Final[tuple[str, ...]] = (
    "career",
    "credibility",
    "behavioral",
    "logistics",
)
_SEMANTIC_SLICES: Final[tuple[str, ...]] = ("semantic", "archetype")
_GATE_SLICES: Final[tuple[str, ...]] = ("integrity", "eligibility")
_VECTOR_SLICES: Final[tuple[str, ...]] = ("quality",)
_ALL_SLICES: Final[tuple[str, ...]] = (
    *_FEATURE_SLICES,
    *_SEMANTIC_SLICES,
    *_GATE_SLICES,
    *_VECTOR_SLICES,
)

# Cumulative set of slices that MUST be populated at each representation stage.
_STAGE_PRESENT: Final[dict[BuildStage, frozenset[str]]] = {
    BuildStage.PARSED: frozenset(),
    BuildStage.FEATURED: frozenset(_FEATURE_SLICES),
    BuildStage.SITUATED: frozenset((*_FEATURE_SLICES, *_SEMANTIC_SLICES)),
    BuildStage.GATED: frozenset((*_FEATURE_SLICES, *_SEMANTIC_SLICES, *_GATE_SLICES)),
    BuildStage.VECTORIZED: frozenset(_ALL_SLICES),
}


@final
class CandidateRepresentation(BaseModel):
    """The universal carrier; engines read slices and return an enriched copy."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    identity: Identity
    career: CareerProfile | None
    credibility: CredibilityProfile | None
    behavioral: BehavioralProfile | None
    logistics: LogisticsProfile | None
    semantic: SemanticProfile | None
    archetype: ArchetypeAssignment | None
    integrity: IntegrityReport | None
    eligibility: EligibilityReport | None
    quality: CandidateQualityVector | None
    stage: BuildStage

    # ----------------------------------------------------------------- invariant
    @model_validator(mode="after")
    def _stage_matches_slices(self) -> CandidateRepresentation:
        expected = _STAGE_PRESENT.get(self.stage)
        if expected is None:
            raise RepresentationStageError(
                f"representation stage {self.stage.name} is beyond VECTORIZED"
            )
        for name in _ALL_SLICES:
            populated = getattr(self, name) is not None
            required = name in expected
            if populated and not required:
                raise RepresentationStageError(
                    f"slice '{name}' populated before stage {self.stage.name} allows"
                )
            if required and not populated:
                raise RepresentationStageError(
                    f"slice '{name}' missing at stage {self.stage.name}"
                )
        return self

    # ------------------------------------------------------------------ factory
    @classmethod
    def parsed(cls, identity: Identity) -> CandidateRepresentation:
        """Construct the initial PARSED-stage representation."""
        return cls(
            identity=identity,
            career=None,
            credibility=None,
            behavioral=None,
            logistics=None,
            semantic=None,
            archetype=None,
            integrity=None,
            eligibility=None,
            quality=None,
            stage=BuildStage.PARSED,
        )

    # --------------------------------------------------------- copy-on-write API
    def _require_stage(self, expected: BuildStage, attaching: str) -> None:
        if self.stage is not expected:
            raise RepresentationStageError(
                f"cannot attach {attaching} at stage {self.stage.name}; "
                f"requires {expected.name}"
            )

    def with_features(
        self,
        *,
        career: CareerProfile,
        credibility: CredibilityProfile,
        behavioral: BehavioralProfile,
        logistics: LogisticsProfile,
    ) -> CandidateRepresentation:
        """Attach the R2 feature slices, advancing PARSED -> FEATURED."""
        self._require_stage(BuildStage.PARSED, "features")
        return CandidateRepresentation(
            identity=self.identity,
            career=career,
            credibility=credibility,
            behavioral=behavioral,
            logistics=logistics,
            semantic=None,
            archetype=None,
            integrity=None,
            eligibility=None,
            quality=None,
            stage=BuildStage.FEATURED,
        )

    def with_semantic(
        self, *, semantic: SemanticProfile, archetype: ArchetypeAssignment
    ) -> CandidateRepresentation:
        """Attach the R3 semantic slices, advancing FEATURED -> SITUATED."""
        self._require_stage(BuildStage.FEATURED, "semantic")
        return CandidateRepresentation(
            identity=self.identity,
            career=self.career,
            credibility=self.credibility,
            behavioral=self.behavioral,
            logistics=self.logistics,
            semantic=semantic,
            archetype=archetype,
            integrity=None,
            eligibility=None,
            quality=None,
            stage=BuildStage.SITUATED,
        )

    def with_gates(
        self, *, integrity: IntegrityReport, eligibility: EligibilityReport
    ) -> CandidateRepresentation:
        """Attach the R4 gate slices, advancing SITUATED -> GATED."""
        self._require_stage(BuildStage.SITUATED, "gates")
        return CandidateRepresentation(
            identity=self.identity,
            career=self.career,
            credibility=self.credibility,
            behavioral=self.behavioral,
            logistics=self.logistics,
            semantic=self.semantic,
            archetype=self.archetype,
            integrity=integrity,
            eligibility=eligibility,
            quality=None,
            stage=BuildStage.GATED,
        )

    def with_quality(self, quality: CandidateQualityVector) -> CandidateRepresentation:
        """Attach the R5 CQV, advancing GATED -> VECTORIZED."""
        self._require_stage(BuildStage.GATED, "quality")
        return CandidateRepresentation(
            identity=self.identity,
            career=self.career,
            credibility=self.credibility,
            behavioral=self.behavioral,
            logistics=self.logistics,
            semantic=self.semantic,
            archetype=self.archetype,
            integrity=self.integrity,
            eligibility=self.eligibility,
            quality=quality,
            stage=BuildStage.VECTORIZED,
        )

    # -------------------------------------------------------- staged accessors
    @property
    def candidate_id(self) -> CandidateId:
        return self.identity.candidate_id

    def require_raw(self) -> RawCandidate:
        """Return the inline ``RawCandidate`` (top-K); raise if only indexed."""
        inline = self.identity.provenance.inline
        if inline is None:
            raise RepresentationStageError(
                "RawCandidate not inlined; re-hydrate before reasoning (R7)"
            )
        return inline

    def require_career(self) -> CareerProfile:
        if self.career is None:
            raise RepresentationStageError("career not populated")
        return self.career

    def require_credibility(self) -> CredibilityProfile:
        if self.credibility is None:
            raise RepresentationStageError("credibility not populated")
        return self.credibility

    def require_behavioral(self) -> BehavioralProfile:
        if self.behavioral is None:
            raise RepresentationStageError("behavioral not populated")
        return self.behavioral

    def require_logistics(self) -> LogisticsProfile:
        if self.logistics is None:
            raise RepresentationStageError("logistics not populated")
        return self.logistics

    def require_semantic(self) -> SemanticProfile:
        if self.semantic is None:
            raise RepresentationStageError("semantic not populated")
        return self.semantic

    def require_archetype(self) -> ArchetypeAssignment:
        if self.archetype is None:
            raise RepresentationStageError("archetype not populated")
        return self.archetype

    def require_integrity(self) -> IntegrityReport:
        if self.integrity is None:
            raise RepresentationStageError("integrity not populated")
        return self.integrity

    def require_eligibility(self) -> EligibilityReport:
        if self.eligibility is None:
            raise RepresentationStageError("eligibility not populated")
        return self.eligibility

    def require_quality(self) -> CandidateQualityVector:
        if self.quality is None:
            raise RepresentationStageError("quality not populated")
        return self.quality

    # --------------------------------------------------------- identity equality
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidateRepresentation):
            return NotImplemented
        return self.identity.candidate_id == other.identity.candidate_id

    def __hash__(self) -> int:
        return hash(self.identity.candidate_id)


__all__: tuple[str, ...] = ("CandidateRepresentation",)
