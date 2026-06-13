"""Closed vocabularies as enums; serialize by value, never by name.

Owner layer: domain.
Allowed imports: stdlib enum.

Ordered enums expose an explicit ``ordinal`` and total ordering so ``<``/``>``
is part of the contract rather than an accident of definition order. Members
never compare across distinct enum types.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class OrderedStrEnum(StrEnum):
    """String enum with an explicit ordinal and a total order.

    ``value`` equals the dataset string exactly (schema-mirroring); ordering
    follows definition order, exposed via ``ordinal``. Comparisons against a
    different enum type return ``NotImplemented`` so cross-type ``<`` raises.
    """

    @property
    def ordinal(self) -> int:
        return list(type(self)).index(self)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.ordinal < other.ordinal
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.ordinal <= other.ordinal
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.ordinal > other.ordinal
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.ordinal >= other.ordinal
        return NotImplemented


# --------------------------------------------------------------------------- #
# Schema-mirroring enums (values equal the dataset strings exactly).
# --------------------------------------------------------------------------- #
class CompanySize(OrderedStrEnum):
    """Company headcount bucket (ordered smallest -> largest)."""

    S_1_10 = "1-10"
    S_11_50 = "11-50"
    S_51_200 = "51-200"
    S_201_500 = "201-500"
    S_501_1000 = "501-1000"
    S_1001_5000 = "1001-5000"
    S_5001_10000 = "5001-10000"
    S_10001_PLUS = "10001+"


class InstitutionTier(OrderedStrEnum):
    """Education institution tier (ordered; ``TIER_1`` strongest, lowest ordinal)."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    UNKNOWN = "unknown"


class WorkMode(StrEnum):
    """Preferred work mode."""

    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    FLEXIBLE = "flexible"


class Proficiency(OrderedStrEnum):
    """Skill proficiency (ordered ascending)."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LanguageProficiency(OrderedStrEnum):
    """Spoken-language proficiency (ordered ascending)."""

    BASIC = "basic"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    NATIVE = "native"


# --------------------------------------------------------------------------- #
# Domain judgement enums.
# --------------------------------------------------------------------------- #
class RelevanceTier(IntEnum):
    """Ground-truth relevance tiering (0..4); honeypots are forced to tier 0."""

    TIER_0 = 0
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3
    TIER_4 = 4


class CareerTrack(StrEnum):
    """Product-vs-services career track (a load-bearing JD signal)."""

    PRODUCT = "product"
    SERVICES = "services"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    """Finding severity."""

    HARD = "hard"
    SOFT = "soft"
    INFO = "info"


class BuildStage(IntEnum):
    """Aggregate maturity discriminant; advances monotonically R1->R7."""

    PARSED = 0
    FEATURED = 1
    SITUATED = 2
    GATED = 3
    VECTORIZED = 4
    SCORED = 5
    RANKED = 6
    EXPLAINED = 7

    @property
    def ordinal(self) -> int:
        return int(self.value)


class ScoreComponent(StrEnum):
    """Closed set of base-relevance components.

    Behavioral and logistics are multipliers, deliberately not components.
    """

    SKILL_MATCH = "skill_match"
    SEMANTIC_FIT = "semantic_fit"
    CAREER_FIT = "career_fit"
    EXPERIENCE_FIT = "experience_fit"
    EDUCATION_FIT = "education_fit"
    CREDIBILITY = "credibility"
    ARCHETYPE_FIT = "archetype_fit"


class IntegrityFlag(StrEnum):
    """Honeypot / impossible-profile rule codes (thresholds calibrated in O3)."""

    TENURE_EXCEEDS_EXPERIENCE = "tenure_exceeds_experience"
    ROLE_DURATION_DATE_MISMATCH = "role_duration_date_mismatch"
    CURRENT_ROLE_HAS_END_DATE = "current_role_has_end_date"
    EXPERT_SKILL_ZERO_USAGE = "expert_skill_zero_usage"
    EDUCATION_TIMELINE_IMPOSSIBLE = "education_timeline_impossible"
    EXPERIENCE_PREDATES_PLAUSIBLE_START = "experience_predates_plausible_start"
    ASSESSMENT_FOR_ABSENT_SKILL = "assessment_for_absent_skill"


class EligibilityCode(StrEnum):
    """JD hard disqualifier / soft penalty codes."""

    PURE_RESEARCH_NO_PRODUCTION = "pure_research_no_production"
    LANGCHAIN_OPENAI_ONLY_RECENT = "langchain_openai_only_recent"
    NO_PRODUCTION_CODE_18M = "no_production_code_18m"
    CONSULTING_FIRMS_ONLY_CAREER = "consulting_firms_only_career"
    PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP = "primary_cv_speech_robotics_no_nlp"
    CLOSED_SOURCE_5Y_NO_VALIDATION = "closed_source_5y_no_validation"
    TITLE_CHASER_SUB_18M_HOPS = "title_chaser_sub_18m_hops"
    NOTICE_OVER_30 = "notice_over_30"
    OUTSIDE_INDIA_NO_SPONSOR = "outside_india_no_sponsor"
    OUTSIDE_EXPERIENCE_BAND = "outside_experience_band"


class ReasoningPolarity(StrEnum):
    """Reasoning clause polarity."""

    STRENGTH = "strength"
    CONCERN = "concern"
    CONTEXT = "context"


class LocationFit(StrEnum):
    """Location fit vs the JD hub set."""

    PREFERRED_HUB = "preferred_hub"
    INDIA_RELOCATABLE = "india_relocatable"
    INDIA_NON_RELOCATABLE = "india_non_relocatable"
    OUTSIDE_INDIA_NO_SPONSOR = "outside_india_no_sponsor"


class NoticeFit(StrEnum):
    """Notice-period fit band."""

    SUB_30_IDEAL = "sub_30_ideal"
    BUYOUTABLE = "buyoutable"
    OVER_30_HIGHER_BAR = "over_30_higher_bar"


class SignalAvailability(StrEnum):
    """Sentinel discriminator distinguishing UNKNOWN from a genuine low value."""

    PRESENT = "present"
    UNKNOWN = "unknown"


class ValidationCode(StrEnum):
    """``validate_submission.py`` codes plus Stage-4 reasoning-quality checks."""

    WRONG_ROW_COUNT = "wrong_row_count"
    RANK_OUT_OF_RANGE = "rank_out_of_range"
    DUPLICATE_RANK = "duplicate_rank"
    MISSING_RANK = "missing_rank"
    DUPLICATE_ID = "duplicate_id"
    BAD_ID_FORMAT = "bad_id_format"
    SCORE_NOT_FLOAT = "score_not_float"
    SCORE_INCREASING = "score_increasing"
    TIEBREAK_VIOLATION = "tiebreak_violation"
    EMPTY_REASONING = "empty_reasoning"
    IDENTICAL_REASONING = "identical_reasoning"
    TEMPLATED_REASONING = "templated_reasoning"
    HALLUCINATION = "hallucination"
    RANK_REASONING_MISMATCH = "rank_reasoning_mismatch"


class EvidenceKind(StrEnum):
    """Provenance evidence kind."""

    PROFILE_FIELD = "profile_field"
    CAREER_FIELD = "career_field"
    SKILL = "skill"
    EDUCATION = "education"
    SIGNAL = "signal"
    DERIVED = "derived"


__all__: tuple[str, ...] = (
    "BuildStage",
    "CareerTrack",
    "CompanySize",
    "EligibilityCode",
    "EvidenceKind",
    "InstitutionTier",
    "IntegrityFlag",
    "LanguageProficiency",
    "LocationFit",
    "NoticeFit",
    "OrderedStrEnum",
    "Proficiency",
    "ReasoningPolarity",
    "RelevanceTier",
    "ScoreComponent",
    "Severity",
    "SignalAvailability",
    "ValidationCode",
    "WorkMode",
)
