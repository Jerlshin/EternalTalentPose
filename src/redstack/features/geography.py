
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import final

from pydantic import BaseModel, ConfigDict, Field

from redstack.domain.candidate.logistics import LogisticsProfile
from redstack.domain.enums import EvidenceKind, LocationFit, NoticeFit
from redstack.domain.ids import LpaAmount
from redstack.domain.source import RawCandidate
from redstack.features.normalize import normalize_text
from redstack.features.parsing import (
    FeatureCell,
    FeatureId,
    feature_id,
    make_cell,
    mint_evidence,
)

__feature_version__ = "1.1.0"

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)

GEO_HUB_MATCH: FeatureId = feature_id("geo", "hub_match")
GEO_INDIA_RELOCATABLE: FeatureId = feature_id("geo", "india_relocatable")
GEO_OUTSIDE_INDIA_NO_SPONSOR: FeatureId = feature_id("geo", "outside_india_no_sponsor")
RELOC_WILLING: FeatureId = feature_id("reloc", "willing")
RELOC_NEEDED: FeatureId = feature_id("reloc", "needed")
NOTICE_FIT: FeatureId = feature_id("notice", "fit")
NOTICE_OVER_30: FeatureId = feature_id("notice", "over_30")
SAL_FIT: FeatureId = feature_id("sal", "fit")
SAL_IS_INVERTED: FeatureId = feature_id("sal", "is_inverted")

# The JD's preferred hubs (Pune / Noida / Hyderabad / Mumbai / Delhi-NCR),
# normalized. Delhi-NCR expands to its constituent cities so a Gurgaon/Gurugram
# or Noida profile matches the NCR hub.
DEFAULT_JD_HUBS: frozenset[str] = frozenset(
    {
        "pune",
        "noida",
        "hyderabad",
        "mumbai",
        "delhi",
        "new delhi",
        "delhi ncr",
        "ncr",
        "gurgaon",
        "gurugram",
        "ghaziabad",
        "faridabad",
        "greater noida",
    }
)

# ``NoticeFit`` -> fit ``UnitScore`` (≤30 ideal, buyout window, >30 higher bar).
_NOTICE_FIT_SCORE: Mapping[NoticeFit, float] = MappingProxyType(
    {
        NoticeFit.SUB_30_IDEAL: 1.0,
        NoticeFit.BUYOUTABLE: 0.6,
        NoticeFit.OVER_30_HIGHER_BAR: 0.25,
    }
)

_NOTICE_IDEAL_MAX_DAYS = 30


@final
class SalaryTarget(BaseModel):
    """The JD's target compensation band (INR lpa), if any."""

    model_config = _STRICT

    min_lpa: LpaAmount = Field(ge=0.0, allow_inf_nan=False)
    max_lpa: LpaAmount = Field(ge=0.0, allow_inf_nan=False)


def _hub_match_value(logistics: LogisticsProfile, jd_hubs: frozenset[str]) -> float:
    """1.0 when the candidate's city is a JD hub.

    Authoritative signal is the pre-derived ``LocationFit.PREFERRED_HUB``; the
    injected ``jd_hubs`` set is a corroborating cross-check on the normalized
    city string (so an updated hub set is honoured without re-deriving the enum).
    ``location`` is consistently "<City>, <State>" in this dataset and
    ``jd_hubs`` holds bare city names, so the full-string check alone never
    matches a hub resident — also check the substring before the first comma.
    """
    if logistics.location_fit is LocationFit.PREFERRED_HUB:
        return 1.0
    normalized = normalize_text(logistics.location)
    city_only = normalized.split(",", 1)[0].strip()
    return 1.0 if normalized in jd_hubs or city_only in jd_hubs else 0.0


def _salary_overlap(
    candidate_min: float,
    candidate_max: float,
    target: SalaryTarget,
) -> float:
    """Jaccard-style overlap of two bands in ``[0, 1]``.

    Inverted candidate bands are first normalized to ``[lo, hi]`` so a sanity
    inversion does not corrupt the fit score (inversion is reported separately).
    """
    lo = min(candidate_min, candidate_max)
    hi = max(candidate_min, candidate_max)
    overlap = max(0.0, min(hi, target.max_lpa) - max(lo, target.min_lpa))
    union = max(hi, target.max_lpa) - min(lo, target.min_lpa)
    if union <= 0.0:
        # Both bands collapse to a point; fit iff they coincide.
        return 1.0 if lo == target.min_lpa else 0.0
    return overlap / union


def extract_geography(
    raw: RawCandidate,
    logistics: LogisticsProfile,
    jd_hubs: frozenset[str] = DEFAULT_JD_HUBS,
    jd_salary: SalaryTarget | None = None,
) -> Mapping[FeatureId, FeatureCell]:
    """Extract the ``geo.* / reloc.* / notice.* / sal.*`` cells for one candidate.

    Deterministic and total. Confidence drops only when the city is unrecognized
    (``LocationFit`` resolved to a non-hub bucket on an unknown city) or when no
    JD salary target is supplied.
    """
    cells: dict[FeatureId, FeatureCell] = {}

    loc_ev = mint_evidence(raw, EvidenceKind.PROFILE_FIELD, "profile.location")
    country_ev = mint_evidence(raw, EvidenceKind.PROFILE_FIELD, "profile.country")
    reloc_ev = mint_evidence(
        raw, EvidenceKind.SIGNAL, "redrob_signals.willing_to_relocate"
    )
    fit = logistics.location_fit
    city_known = fit is not LocationFit.INDIA_NON_RELOCATABLE or logistics.willing_to_relocate
    geo_conf = 0.95 if fit is LocationFit.PREFERRED_HUB else (0.85 if city_known else 0.5)

    # --- geo.* -------------------------------------------------------------- #
    cells[GEO_HUB_MATCH] = make_cell(
        _hub_match_value(logistics, jd_hubs), geo_conf, (loc_ev,)
    )
    india_relocatable = (
        fit is LocationFit.INDIA_RELOCATABLE
        or (fit is LocationFit.PREFERRED_HUB)
        or (fit is LocationFit.INDIA_NON_RELOCATABLE and logistics.willing_to_relocate)
    )
    cells[GEO_INDIA_RELOCATABLE] = make_cell(
        1.0 if india_relocatable else 0.0, geo_conf, (loc_ev, reloc_ev)
    )
    cells[GEO_OUTSIDE_INDIA_NO_SPONSOR] = make_cell(
        1.0 if fit is LocationFit.OUTSIDE_INDIA_NO_SPONSOR else 0.0,
        geo_conf,
        (country_ev,),
    )

    # --- reloc.* ------------------------------------------------------------ #
    cells[RELOC_WILLING] = make_cell(
        1.0 if logistics.willing_to_relocate else 0.0, 0.95, (reloc_ev,)
    )
    # "needed" iff not already in a preferred hub.
    needed = fit is not LocationFit.PREFERRED_HUB
    cells[RELOC_NEEDED] = make_cell(1.0 if needed else 0.0, geo_conf, (loc_ev,))

    # --- notice.* ----------------------------------------------------------- #
    notice_ev = mint_evidence(
        raw, EvidenceKind.SIGNAL, "redrob_signals.notice_period_days"
    )
    cells[NOTICE_FIT] = make_cell(
        _NOTICE_FIT_SCORE[logistics.notice_fit], 0.95, (notice_ev,)
    )
    cells[NOTICE_OVER_30] = make_cell(
        1.0 if logistics.notice_period_days > _NOTICE_IDEAL_MAX_DAYS else 0.0,
        0.95,
        (notice_ev,),
    )

    # --- sal.* -------------------------------------------------------------- #
    sal_min_ev = mint_evidence(
        raw, EvidenceKind.SIGNAL,
        "redrob_signals.expected_salary_range_inr_lpa.min",
    )
    sal_max_ev = mint_evidence(
        raw, EvidenceKind.SIGNAL,
        "redrob_signals.expected_salary_range_inr_lpa.max",
    )
    if jd_salary is not None:
        sal_fit = _salary_overlap(
            float(logistics.salary.min_lpa),
            float(logistics.salary.max_lpa),
            jd_salary,
        )
        sal_conf = 0.7
    else:
        # No JD band: neutral fit, reduced confidence (cannot judge alignment).
        sal_fit = 0.5
        sal_conf = 0.3
    cells[SAL_FIT] = make_cell(sal_fit, sal_conf, (sal_min_ev, sal_max_ev))
    cells[SAL_IS_INVERTED] = make_cell(
        1.0 if logistics.salary.is_inverted else 0.0, 0.9, (sal_min_ev, sal_max_ev)
    )

    return MappingProxyType(cells)


__all__: tuple[str, ...] = (
    "DEFAULT_JD_HUBS",
    "GEO_HUB_MATCH",
    "GEO_INDIA_RELOCATABLE",
    "GEO_OUTSIDE_INDIA_NO_SPONSOR",
    "NOTICE_FIT",
    "NOTICE_OVER_30",
    "RELOC_NEEDED",
    "RELOC_WILLING",
    "SAL_FIT",
    "SAL_IS_INVERTED",
    "SalaryTarget",
    "extract_geography",
)