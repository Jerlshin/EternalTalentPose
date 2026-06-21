from __future__ import annotations

from datetime import date
from typing import Final, final

from pydantic import BaseModel, ConfigDict

from redstack.config.schema import IntegrityThresholds
from redstack.domain.candidate.career import CareerProfile, PositionFact
from redstack.domain.candidate.integrity import IntegrityFinding, IntegrityReport
from redstack.domain.enums import EvidenceKind, IntegrityFlag, Proficiency, Severity
from redstack.domain.errors import ProvenanceError
from redstack.domain.ids import UnitScore
from redstack.domain.provenance import EvidenceRef
from redstack.domain.source import RawCandidate
from redstack.features.view import clamp_unit, days_between, make_evidence

# All seven flags are categorical impossibilities → HARD unless calibration
# overrides per flag. Re-stated here as the closed evaluated set (determinism).
_ALL_FLAGS: Final[frozenset[IntegrityFlag]] = frozenset(IntegrityFlag)

_DAYS_PER_MONTH: Final[float] = 30.4375


@final
class IntegrityEngine(BaseModel):
    """Stateless, pure honeypot/contradiction engine.

    Construction takes the calibrated ``IntegrityThresholds`` only; the engine
    holds no mutable state and performs no IO. ``evaluate`` is a pure function of
    ``(CareerProfile, RawCandidate)`` given those thresholds.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=False
    )

    thresholds: IntegrityThresholds

    # ------------------------------------------------------------------ public
    def evaluate(self, career: CareerProfile, raw: RawCandidate) -> IntegrityReport:
        """Run all seven rules, aggregate risk, finalize the ``IntegrityReport``."""
        findings: list[IntegrityFinding] = []
        findings.extend(self._tenure_exceeds_experience(career, raw))
        findings.extend(self._role_duration_date_mismatch(career, raw))
        findings.extend(self._current_role_has_end_date(career, raw))
        findings.extend(self._expert_skill_zero_usage(raw))
        findings.extend(self._education_timeline_impossible(raw))
        findings.extend(self._experience_predates_plausible_start(career, raw))
        findings.extend(self._assessment_for_absent_skill(raw))

        ordered = tuple(sorted(findings, key=lambda f: f.code.value))
        honeypot_score = self._aggregate(ordered)
        hard_count = sum(1 for f in ordered if f.severity is Severity.HARD)
        is_honeypot = (hard_count >= 2) or (
            float(honeypot_score) >= self.thresholds.honeypot_threshold
        )
        return IntegrityReport(
            findings=ordered,
            honeypot_score=honeypot_score,
            is_honeypot=is_honeypot,
            rules_evaluated=_ALL_FLAGS,
        )

    # --------------------------------------------------------------- internals
    def _severity_of(self, flag: IntegrityFlag) -> Severity:
        return self.thresholds.flag_severity.get(flag, Severity.HARD)

    def _aggregate(self, findings: tuple[IntegrityFinding, ...]) -> UnitScore:
        """Calibrated weighted aggregate of fired flags → ``[0, 1]``.

        Weights are calibrated (O3) so that no single flag crosses the honeypot
        threshold alone, while two corroborating impossibilities do. Determinism:
        summed in sorted-code order, clamped into the unit interval.
        """
        weights = self.thresholds.flag_weights
        total = 0.0
        for finding in findings:
            total += weights.get(finding.code, 0.0)
        return UnitScore(clamp_unit(total))

    # -- rule 1 -------------------------------------------------------------- #
    def _tenure_exceeds_experience(
        self, career: CareerProfile, raw: RawCandidate
    ) -> tuple[IntegrityFinding, ...]:
        summed_months = sum(int(p.duration_months) for p in career.positions)
        summed_years = summed_months / 12.0
        stated = float(raw.profile.years_of_experience)
        if summed_years <= stated + self.thresholds.tolerance_experience_years:
            return ()
        flag = IntegrityFlag.TENURE_EXCEEDS_EXPERIENCE
        evidence = (
            make_evidence(
                EvidenceKind.DERIVED,
                "career.summed_duration_years",
                round(summed_years, 4),
            ),
            make_evidence(
                EvidenceKind.PROFILE_FIELD,
                "profile.years_of_experience",
                stated,
                raw=raw,
            ),
        )
        return (
            IntegrityFinding(
                code=flag,
                severity=self._severity_of(flag),
                evidence=evidence,
                detail=(
                    f"summed role tenure {summed_years:.1f}y exceeds stated "
                    f"experience {stated:.1f}y beyond tolerance"
                ),
            ),
        )

    @staticmethod
    def _raw_index(raw: RawCandidate, pos: PositionFact) -> int:
        """The ``raw.career_history`` index matching a ``PositionFact``.

        ``CareerProfile.positions`` is reverse-chronologically re-sorted by
        :func:`build_career_profile`, so its index does not generally line up
        with ``raw.career_history``'s -- citing ``positions``' enumeration
        index directly would mint evidence pointing at an unrelated raw
        record. ``(company, start_date)`` is copied verbatim into
        ``PositionFact`` and is therefore a reliable join key back to raw.
        """
        for index, candidate in enumerate(raw.career_history):
            if (
                candidate.company == pos.company
                and candidate.start_date == pos.start_date
            ):
                return index
        raise ProvenanceError(
            f"no career_history entry matches position {pos.company!r}/{pos.start_date}"
        )

    # -- rule 2 -------------------------------------------------------------- #
    def _role_duration_date_mismatch(
        self, career: CareerProfile, raw: RawCandidate
    ) -> tuple[IntegrityFinding, ...]:
        flag = IntegrityFlag.ROLE_DURATION_DATE_MISMATCH
        out: list[IntegrityFinding] = []
        tol = float(self.thresholds.duration_date_tolerance_months)
        for pos in career.positions:
            if pos.end_date is None:
                continue
            span_months = days_between(pos.end_date, pos.start_date) / _DAYS_PER_MONTH
            if abs(span_months - float(pos.duration_months)) <= tol:
                continue
            idx = self._raw_index(raw, pos)
            out.append(
                IntegrityFinding(
                    code=flag,
                    severity=self._severity_of(flag),
                    evidence=(
                        make_evidence(
                            EvidenceKind.CAREER_FIELD,
                            f"career_history[{idx}].duration_months",
                            int(pos.duration_months),
                        ),
                        make_evidence(
                            EvidenceKind.CAREER_FIELD,
                            f"career_history[{idx}].start_date",
                            pos.start_date.isoformat(),
                        ),
                        make_evidence(
                            EvidenceKind.CAREER_FIELD,
                            f"career_history[{idx}].end_date",
                            pos.end_date.isoformat(),
                        ),
                    ),
                    detail=(
                        f"role {idx} duration {int(pos.duration_months)}m vs "
                        f"date span {span_months:.1f}m"
                    ),
                )
            )
        return tuple(out)

    # -- rule 3 -------------------------------------------------------------- #
    def _current_role_has_end_date(
        self, career: CareerProfile, raw: RawCandidate
    ) -> tuple[IntegrityFinding, ...]:
        flag = IntegrityFlag.CURRENT_ROLE_HAS_END_DATE
        out: list[IntegrityFinding] = []
        for pos in career.positions:
            if not (pos.is_current and pos.end_date is not None):
                continue
            idx = self._raw_index(raw, pos)
            out.append(
                IntegrityFinding(
                    code=flag,
                    severity=self._severity_of(flag),
                    evidence=(
                        make_evidence(
                            EvidenceKind.CAREER_FIELD,
                            f"career_history[{idx}].is_current",
                            True,
                        ),
                        make_evidence(
                            EvidenceKind.CAREER_FIELD,
                            f"career_history[{idx}].end_date",
                            pos.end_date.isoformat(),
                        ),
                    ),
                    detail=f"role {idx} marked current yet carries an end_date",
                )
            )
        return tuple(out)

    # -- rule 4 -------------------------------------------------------------- #
    def _expert_skill_zero_usage(
        self, raw: RawCandidate
    ) -> tuple[IntegrityFinding, ...]:
        flag = IntegrityFlag.EXPERT_SKILL_ZERO_USAGE
        offenders: list[tuple[int, str]] = []
        for idx, skill in enumerate(raw.skills):
            advanced = skill.proficiency >= Proficiency.ADVANCED
            zero_usage = (
                skill.duration_months is None or int(skill.duration_months) == 0
            )
            if advanced and zero_usage:
                offenders.append((idx, skill.name))
        if len(offenders) < self.thresholds.expert_zero_usage_min_count:
            return ()
        evidence = tuple(
            make_evidence(EvidenceKind.SKILL, f"skills[{idx}].name", name, raw=raw)
            for idx, name in offenders
        )
        return (
            IntegrityFinding(
                code=flag,
                severity=self._severity_of(flag),
                evidence=evidence,
                detail=(
                    f"{len(offenders)} skills at >=ADVANCED with zero "
                    "months/duration (en-masse expert-without-usage)"
                ),
            ),
        )

    # -- rule 5 -------------------------------------------------------------- #
    def _education_timeline_impossible(
        self, raw: RawCandidate
    ) -> tuple[IntegrityFinding, ...]:
        flag = IntegrityFlag.EDUCATION_TIMELINE_IMPOSSIBLE
        out: list[IntegrityFinding] = []
        for idx, edu in enumerate(raw.education):
            if edu.end_year >= edu.start_year:
                continue
            out.append(
                IntegrityFinding(
                    code=flag,
                    severity=self._severity_of(flag),
                    evidence=(
                        make_evidence(
                            EvidenceKind.EDUCATION,
                            f"education[{idx}].start_year",
                            int(edu.start_year),
                            raw=raw,
                        ),
                        make_evidence(
                            EvidenceKind.EDUCATION,
                            f"education[{idx}].end_year",
                            int(edu.end_year),
                            raw=raw,
                        ),
                    ),
                    detail=(
                        f"education {idx} ends {int(edu.end_year)} before it "
                        f"starts {int(edu.start_year)}"
                    ),
                )
            )
        return tuple(out)

    # -- rule 6 -------------------------------------------------------------- #
    def _experience_predates_plausible_start(
        self, career: CareerProfile, raw: RawCandidate
    ) -> tuple[IntegrityFinding, ...]:
        flag = IntegrityFlag.EXPERIENCE_PREDATES_PLAUSIBLE_START
        if not career.positions:
            return ()
        if not raw.education:
            return ()
        edu_idx = min(
            range(len(raw.education)), key=lambda i: raw.education[i].start_year
        )
        earliest_education = raw.education[edu_idx].start_year
        earliest_position = min(career.positions, key=lambda p: p.start_date)
        earliest_career: date = earliest_position.start_date
        tol = self.thresholds.experience_predates_tolerance_years
        if earliest_career.year >= earliest_education - tol:
            return ()
        career_idx = self._raw_index(raw, earliest_position)
        return (
            IntegrityFinding(
                code=flag,
                severity=self._severity_of(flag),
                evidence=(
                    make_evidence(
                        EvidenceKind.CAREER_FIELD,
                        f"career_history[{career_idx}].start_date",
                        earliest_career.year,
                        raw=raw,
                    ),
                    make_evidence(
                        EvidenceKind.EDUCATION,
                        f"education[{edu_idx}].start_year",
                        earliest_education,
                        raw=raw,
                    ),
                ),
                detail=(
                    f"career begins {earliest_career.year}, implausibly before "
                    f"first education {earliest_education}"
                ),
            ),
        )

    # -- rule 7 -------------------------------------------------------------- #
    def _assessment_for_absent_skill(
        self, raw: RawCandidate
    ) -> tuple[IntegrityFinding, ...]:
        flag = IntegrityFlag.ASSESSMENT_FOR_ABSENT_SKILL
        scores = raw.redrob_signals.skill_assessment_scores
        if not scores:
            return ()
        skill_names = {s.name for s in raw.skills}
        orphans = sorted(name for name in scores if name not in skill_names)
        if not orphans:
            return ()
        evidence = tuple(
            make_evidence(
                EvidenceKind.SIGNAL,
                f'redrob_signals.skill_assessment_scores["{name}"]',
                float(scores[name]),
                raw=raw,
            )
            for name in orphans
        )
        return (
            IntegrityFinding(
                code=flag,
                severity=self._severity_of(flag),
                evidence=evidence,
                detail=(
                    f"{len(orphans)} assessment score(s) reference skills absent "
                    "from the declared skill list"
                ),
            ),
        )


__all__: tuple[str, ...] = ("IntegrityEngine",)
