
from __future__ import annotations

import enum
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from redstack.domain.enums import IntegrityFlag, LocationFit, NoticeFit, Severity

__all__ = [
    "RunMode",
    "Profile",
    "LogLevel",
    "MalformedRecordPolicy",
    "ScoreTransformKind",
    "AnchorPolarity",
    "HoneypotSeverity",
    "DeterminismConfig",
    "BudgetConfig",
    "PathsConfig",
    "LoggingConfig",
    "ScorePresentationConfig",
    "OnlineRuntimeConfig",
    "OfflineRuntimeConfig",
    "RedstackConfig",
    "ScoringWeightsConfig",
    "ConceptSeed",
    "LexiconSeedConfig",
    "AnchorIntent",
    "JdAnchorsConfig",
    "EligibilityRule",
    "EligibilityRulesConfig",
    "HoneypotRule",
    "HoneypotRulesConfig",
    "IntegrityThresholds",
    "EligibilityRuleSet",
    "ScoringPolicy",
    "BehavioralPolicy",
    "LogisticsPolicy",
]


# --------------------------------------------------------------------------- #
# Closed config-layer vocabularies (IO/authoring concepts, not domain VOs).   #
# --------------------------------------------------------------------------- #
class RunMode(enum.StrEnum):
    """Which runtime layer the loader composes (``runtime/<mode>.yaml``)."""

    ONLINE = "online"
    OFFLINE = "offline"


class Profile(enum.StrEnum):
    """The optional final override layer (``profiles/<profile>.yaml``)."""

    CI = "ci"
    LOCAL = "local"


class LogLevel(enum.StrEnum):
    """Structured-logging verbosity (see :mod:`redstack.observability.logging`)."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class MalformedRecordPolicy(enum.StrEnum):
    """Ingestion policy for a ``Malformed`` source record (online R1 / offline O0).

    ``SKIP`` drops the record as data and continues; ``ABORT`` fails the run.
    """

    SKIP = "skip"
    ABORT = "abort"


class ScoreTransformKind(enum.StrEnum):
    """How an internal base score is presented in the submission CSV column."""

    IDENTITY = "identity"


class AnchorPolarity(enum.StrEnum):
    """Authoring polarity of a ``jd.*`` anchor intent (Architecture §13, O6)."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class HoneypotSeverity(enum.StrEnum):
    """Authoring severity of a honeypot rule shape.

    ``HARD`` rules contribute to the ``>= 2 HARD`` honeypot decision; ``SOFT``
    rules feed the composite. O3 calibrates the numeric thresholds against the
    census; this seed only declares the rule *shape*.
    """

    HARD = "hard"
    SOFT = "soft"


# --------------------------------------------------------------------------- #
# Base model: every config VO is frozen and rejects unknown keys.             #
# --------------------------------------------------------------------------- #
class _FrozenConfig(BaseModel):
    """Shared base enforcing the frozen / ``extra="forbid"`` discipline."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Runtime / IO config sections.                                               #
# --------------------------------------------------------------------------- #
class DeterminismConfig(_FrozenConfig):
    """Seed and thread pins owned by :mod:`redstack.config.determinism`.

    ``omp_num_threads`` and ``mkl_num_threads`` default to ``1`` to guarantee
    thread-count-invariant, byte-identical CSV outputs. They are configurable
    only so the determinism test suite can prove 1-thread vs N-thread ranking
    identity; production runs keep them pinned to ``1``.
    """

    seed: int = Field(default=20240601, ge=0)
    omp_num_threads: int = Field(default=1, ge=1)
    mkl_num_threads: int = Field(default=1, ge=1)
    onnx_intra_op_threads: int = Field(default=1, ge=1)
    onnx_inter_op_threads: int = Field(default=1, ge=1)


class BudgetConfig(_FrozenConfig):
    """The hard online runtime budget asserted by the timing guard.

    ``max_wall_seconds`` is the disqualifying ceiling (300 s); the internal
    ``target_wall_seconds`` (130 s) is the regression alarm. ``max_rss_mb`` is
    the 16 GB memory ceiling.
    """

    max_wall_seconds: float = Field(default=300.0, gt=0.0)
    target_wall_seconds: float = Field(default=130.0, gt=0.0)
    max_rss_mb: int = Field(default=16384, gt=0)

    @model_validator(mode="after")
    def _target_within_ceiling(self) -> BudgetConfig:
        """The internal target must not exceed the disqualifying ceiling."""
        if self.target_wall_seconds > self.max_wall_seconds:
            msg = (
                "target_wall_seconds "
                f"({self.target_wall_seconds}) must be <= max_wall_seconds "
                f"({self.max_wall_seconds})"
            )
            raise ValueError(msg)
        return self


class PathsConfig(_FrozenConfig):
    """Filesystem locations. Paths are kept as ``str`` for byte-stable hashing.

    ``configs/`` is intent and ``data/`` is raw fact (read-only); only the
    offline pipeline writes ``artifacts/``. No code writes ``data/`` or
    ``configs/``.
    """

    artifacts_root: str = Field(default="artifacts", min_length=1)
    data_root: str = Field(default="data", min_length=1)
    candidates_path: str = Field(default="data/raw/candidates.jsonl", min_length=1)
    golden_labels_path: str = Field(
        default="data/golden/golden_labels.csv", min_length=1
    )
    manifest_path: str = Field(default="artifacts/MANIFEST.json", min_length=1)
    submission_path: str = Field(default="submission.csv", min_length=1)
    run_report_path: str = Field(default="run_report.json", min_length=1)


class LoggingConfig(_FrozenConfig):
    """Logging policy. ``deterministic`` strips timestamps from repro-relevant lines."""

    level: LogLevel = LogLevel.INFO
    deterministic: bool = True


class ScorePresentationConfig(_FrozenConfig):
    """How scores reach the CSV: transform, decimal precision, FLOOR sentinel.

    ``floor_sentinel`` is the value assigned to integrity/eligibility-gated
    candidates so a gated record can never outrank a genuine one.
    """

    transform: ScoreTransformKind = ScoreTransformKind.IDENTITY
    decimals: int = Field(default=6, ge=0, le=12)
    floor_sentinel: float = Field(default=0.0)


class OnlineRuntimeConfig(_FrozenConfig):
    """Online-only knobs (``runtime/online.yaml``), consumed at R0.

    Holds the injected ``as_of`` clock (no module reads the OS clock for logic),
    the top-K cut, the malformed-record policy, and the score presentation.
    """

    as_of: datetime
    top_k: int = Field(default=100, ge=1)
    malformed_record_policy: MalformedRecordPolicy = MalformedRecordPolicy.SKIP
    score_presentation: ScorePresentationConfig = Field(
        default_factory=ScorePresentationConfig
    )

    @field_validator("as_of")
    @classmethod
    def _as_of_must_be_aware(cls, value: datetime) -> datetime:
        """Reject naive datetimes; ``as_of`` must carry an explicit timezone."""
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            msg = "online.as_of must be timezone-aware (include an explicit offset)"
            raise ValueError(msg)
        return value


class OfflineRuntimeConfig(_FrozenConfig):
    """Offline build parameters (``runtime/offline.yaml``).

    Holds the injected ``as_of`` clock (so recency-dependent feature
    extraction in O14 is byte-stable across rebuilds, mirroring why
    :class:`OnlineRuntimeConfig` injects its own), the offline RNG seed, the
    pinned SentenceTransformer model id and revision, the KMeans ``k`` for
    archetype discovery (O7), the listwise weight-search budget (O9), and
    batch sizes for embedding/feature passes.
    """

    as_of: datetime
    seed: int = Field(default=20240601, ge=0)
    st_model_id: str = Field(min_length=1)
    st_model_revision: str = Field(min_length=1)
    kmeans_k: int = Field(ge=1)
    search_budget: int = Field(ge=1)
    embedding_batch_size: int = Field(default=256, ge=1)
    feature_batch_size: int = Field(default=4096, ge=1)

    @field_validator("as_of")
    @classmethod
    def _as_of_must_be_aware(cls, value: datetime) -> datetime:
        """Reject naive datetimes; ``as_of`` must carry an explicit timezone."""
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            msg = "offline.as_of must be timezone-aware (include an explicit offset)"
            raise ValueError(msg)
        return value


class RedstackConfig(_FrozenConfig):
    """The fully-composed, validated runtime configuration.

    Produced by :func:`redstack.config.loader.load_config` from the deterministic
    deep-merge of ``base -> runtime/<mode> -> profiles/<profile>``. Exactly one
    of :attr:`online` / :attr:`offline` is populated, matching :attr:`run_mode`.
    This is the typed object every pipeline and CLI verb receives; raw ``dict``
    config never crosses a module boundary.
    """

    schema_version: str = Field(min_length=1)
    run_mode: RunMode
    profile: Profile | None = None
    determinism: DeterminismConfig = Field(default_factory=DeterminismConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    online: OnlineRuntimeConfig | None = None
    offline: OfflineRuntimeConfig | None = None

    @model_validator(mode="after")
    def _mode_block_consistency(self) -> RedstackConfig:
        """Exactly the run-mode's block must be present; the other must be absent."""
        if self.run_mode is RunMode.ONLINE:
            if self.online is None:
                raise ValueError("run_mode=online requires an 'online' block")
            if self.offline is not None:
                raise ValueError("run_mode=online forbids an 'offline' block")
        else:
            if self.offline is None:
                raise ValueError("run_mode=offline requires an 'offline' block")
            if self.online is not None:
                raise ValueError("run_mode=offline forbids an 'online' block")
        return self


# --------------------------------------------------------------------------- #
# Behaviour / authoring seeds (consumed by the offline pipeline).             #
# --------------------------------------------------------------------------- #
class ScoringWeightsConfig(_FrozenConfig):
    """Seed/candidate scoring weights — the O9 search *input*, not the contract.

    Keys are score-component identifiers (mapped to ``domain.ScoreComponent`` at
    the O9/artifact boundary). The frozen, validated weights consumed online
    live in ``artifacts/weights/scoring_weights.locked.yaml``, never here, which
    prevents "I tweaked the YAML and the leaderboard moved" drift.
    """

    layout_version: str = Field(min_length=1)
    weights: dict[str, float] = Field(min_length=1)

    @field_validator("weights")
    @classmethod
    def _weights_finite(cls, value: dict[str, float]) -> dict[str, float]:
        """Every seed weight must be a finite real number."""
        import math

        for component, weight in value.items():
            if not math.isfinite(weight):
                msg = f"scoring weight for {component!r} is not finite: {weight!r}"
                raise ValueError(msg)
        return value


class ConceptSeed(_FrozenConfig):
    """Human seed terms for one lexicon concept (mined by O4, expanded by O5)."""

    seed_terms: tuple[str, ...] = Field(min_length=1)

    @field_validator("seed_terms")
    @classmethod
    def _terms_non_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Seed terms must be non-blank after stripping."""
        if any(not term.strip() for term in value):
            raise ValueError("lexicon seed_terms must not contain blank entries")
        return value


class LexiconSeedConfig(_FrozenConfig):
    """The ``lexicon/lexicon.seed.yaml`` authoring seed: concept -> seed terms."""

    concepts: dict[str, ConceptSeed] = Field(min_length=1)


class AnchorIntent(_FrozenConfig):
    """One ``jd.*`` anchor intent: a stable id, polarity, and the anchor text.

    The text is embedded offline (O6) into ``anchor_vectors.npy``; the id is the
    key the online ``SemanticEngine`` resolves cosine similarity against.
    """

    id: str = Field(min_length=1)
    polarity: AnchorPolarity
    text: str = Field(min_length=1)


class JdAnchorsConfig(_FrozenConfig):
    """The ``anchors/jd_anchors.yaml`` authoring seed (re-authored per JD)."""

    anchors: tuple[AnchorIntent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique_and_polarities_present(self) -> JdAnchorsConfig:
        """Anchor ids must be unique and both polarities must be represented."""
        ids = [anchor.id for anchor in self.anchors]
        if len(set(ids)) != len(ids):
            raise ValueError("jd anchor ids must be unique")
        polarities = {anchor.polarity for anchor in self.anchors}
        if AnchorPolarity.POSITIVE not in polarities:
            raise ValueError("jd anchors require at least one positive intent")
        if AnchorPolarity.NEGATIVE not in polarities:
            raise ValueError("jd anchors require at least one negative intent")
        return self


class EligibilityRule(_FrozenConfig):
    """A declarative JD eligibility rule keyed by an eligibility code.

    The code string is cross-checked against ``domain.EligibilityCode`` when O6
    packages this seed into the artifact (Repository Layout §5). ``penalty`` is
    populated for soft penalties (in ``[0, 1]``) and omitted for hard blocks.
    """

    code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    penalty: float | None = Field(default=None, ge=0.0, le=1.0)


class EligibilityRulesConfig(_FrozenConfig):
    """The ``gates/eligibility_rules.yaml`` authoring seed.

    Hard blocks carry no penalty (they disqualify); soft penalties must each
    declare a bounded penalty weight.
    """

    hard_blocks: tuple[EligibilityRule, ...] = Field(default=())
    soft_penalties: tuple[EligibilityRule, ...] = Field(default=())

    @model_validator(mode="after")
    def _shape_and_uniqueness(self) -> EligibilityRulesConfig:
        """Hard blocks omit penalties; soft penalties require them; codes unique."""
        for rule in self.hard_blocks:
            if rule.penalty is not None:
                raise ValueError(f"hard block {rule.code!r} must not carry a penalty")
        for rule in self.soft_penalties:
            if rule.penalty is None:
                raise ValueError(
                    f"soft penalty {rule.code!r} must declare a penalty weight"
                )
        codes = [r.code for r in (*self.hard_blocks, *self.soft_penalties)]
        if len(set(codes)) != len(codes):
            raise ValueError("eligibility rule codes must be unique")
        if not codes:
            raise ValueError("eligibility rules must define at least one rule")
        return self


class HoneypotRule(_FrozenConfig):
    """A human-declared honeypot rule shape keyed by an integrity flag.

    The flag string is cross-checked against ``domain.IntegrityFlag`` at the O3
    artifact boundary. O3 calibrates the numeric threshold against the census;
    this seed declares only the rule's identity and severity.
    """

    flag: str = Field(min_length=1)
    severity: HoneypotSeverity
    description: str = Field(min_length=1)


class HoneypotRulesConfig(_FrozenConfig):
    """The ``integrity/honeypot_rules.yaml`` authoring seed."""

    rules: tuple[HoneypotRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _flags_unique(self) -> HoneypotRulesConfig:
        """Integrity flags must be unique across declared rules."""
        flags = [rule.flag for rule in self.rules]
        if len(set(flags)) != len(flags):
            raise ValueError("honeypot rule flags must be unique")
        return self


# --------------------------------------------------------------------------- #
# Calibrated artifact-shaped policy models — consumed *online* by the engines #
# (sourced from compiled O3/O6/O9/O11 artifacts at R0, never hand-authored).  #
# --------------------------------------------------------------------------- #
class IntegrityThresholds(_FrozenConfig):
    """Calibrated honeypot thresholds (O3 artifact); consumed by ``IntegrityEngine``.

    Sourced from ``artifacts/calibration/integrity_thresholds.json``. Per-flag
    ``flag_severity``/``flag_weights`` default to empty (the engine falls back
    to ``Severity.HARD`` / weight ``0.0`` for any flag absent from the map).
    """

    honeypot_threshold: float = Field(ge=0.0, le=1.0)
    flag_severity: dict[IntegrityFlag, Severity] = Field(default_factory=dict)
    flag_weights: dict[IntegrityFlag, float] = Field(default_factory=dict)
    tolerance_experience_years: float = Field(ge=0.0)
    duration_date_tolerance_months: float = Field(ge=0.0)
    expert_zero_usage_min_count: int = Field(ge=1)
    experience_predates_tolerance_years: int = Field(ge=0)


class EligibilityRuleSet(_FrozenConfig):
    """Calibrated eligibility-gate thresholds (O6 artifact); consumed by
    ``EligibilityEngine``.

    Sourced from the compiled ``gates/eligibility_rules.yaml`` artifact — the
    JD-derived numeric thresholds behind each hard block / soft penalty.
    """

    research_min_semantic_fit: float = Field(ge=0.0, le=1.0)
    framework_only_stuffing_min: float = Field(ge=0.0, le=1.0)
    framework_only_gap_min: float = Field(ge=0.0, le=1.0)
    production_recency_max_months: int = Field(ge=0)
    adjacent_domain_min_relevant_credibility: float = Field(ge=0.0, le=1.0)
    adjacent_domain_min_negative_fit: float = Field(ge=-1.0, le=1.0)
    adjacent_domain_min_skill_match: float = Field(ge=0.0, le=1.0)
    closed_source_min_years: float = Field(ge=0.0)
    closed_source_max_credible_skills: int = Field(ge=0)
    title_chaser_min_hop_rate: float = Field(ge=0.0, le=1.0)
    experience_band_min_years: float = Field(ge=0.0)
    experience_band_max_years: float = Field(ge=0.0)


def default_eligibility_rules() -> EligibilityRuleSet:
    """The locked ``EligibilityRuleSet`` thresholds.

    ``gates/eligibility_rules.yaml`` (O6) carries only human descriptions per
    code -- O6 never calibrates numeric thresholds, it authors rule *shape*
    (no gold-labeled data exists for these). The literal values below are
    therefore the single source of truth both the offline (O13a) and online
    (R0) paths construct their ``EligibilityEngine`` from, so the two stay in
    lockstep by construction rather than by two hand-synced literals.
    """
    return EligibilityRuleSet(
        research_min_semantic_fit=0.5,
        framework_only_stuffing_min=0.6,
        framework_only_gap_min=0.3,
        production_recency_max_months=18,
        # Raised from 0.3: diagnostic evidence across 27 real candidates found
        # a clean gap between domain-irrelevant titles (HR Manager, Civil
        # Engineer, Accountant, ... — 0.11-0.50) and genuinely ML-titled ones
        # (Senior ML Engineer / Applied Scientist / RecSys Engineer —
        # 0.80-0.94); 0.6 sits in that gap with margin on both sides.
        adjacent_domain_min_relevant_credibility=0.6,
        adjacent_domain_min_negative_fit=0.3,
        # Catches the complementary failure mode: a candidate whose few
        # listed skills are all non-ML but well-corroborated (so
        # relevant_skill_credibility alone reads high) still needs *some*
        # claimed ML-specific competency to pass. 0.05 sits comfortably below
        # every genuinely ML-titled sample observed (0.15-0.19+) and at/above
        # the near-zero values non-ML titles showed.
        adjacent_domain_min_skill_match=0.05,
        closed_source_min_years=5.0,
        closed_source_max_credible_skills=0,
        title_chaser_min_hop_rate=0.5,
        experience_band_min_years=2.0,
        experience_band_max_years=15.0,
    )


class ScoringPolicy(_FrozenConfig):
    """Scoring combination policy (floor + neutral prior); consumed by
    ``ScoringEngine``."""

    floor: float = Field(default=0.0)
    neutral_prior: float = Field(default=0.5, ge=0.0, le=1.0)


class BehavioralPolicy(_FrozenConfig):
    """Behavioral-multiplier policy (O11 artifact); consumed by ``BehavioralEngine``."""

    family_weights: dict[str, float] = Field(default_factory=dict)
    unknown_neutral_base: float = Field(default=0.5, ge=0.0, le=1.0)
    m_min: float = Field(ge=0.0, le=1.0)
    m_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bounds_ordered(self) -> BehavioralPolicy:
        if self.m_min > self.m_max:
            raise ValueError("m_min must not exceed m_max")
        return self


class LogisticsPolicy(_FrozenConfig):
    """Logistics-multiplier policy; consumed by ``LogisticsEngine``."""

    location_fit_factor: dict[LocationFit, float] = Field(default_factory=dict)
    location_default_factor: float = Field(default=1.0)
    notice_fit_factor: dict[NoticeFit, float] = Field(default_factory=dict)
    notice_default_factor: float = Field(default=1.0)
    work_mode_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    salary_inversion_factor: float = Field(default=1.0, ge=0.0, le=1.0)
    m_min: float = Field(ge=0.0, le=1.0)
    m_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bounds_ordered(self) -> LogisticsPolicy:
        if self.m_min > self.m_max:
            raise ValueError("m_min must not exceed m_max")
        return self
