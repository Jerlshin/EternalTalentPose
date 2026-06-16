"""O3 — Honeypot Discovery.

Owner stage: O3 (Offline Pipeline Part 4; Feature Layer Part 5; Engine Layer §3
``CandidateConsistencyEngine`` in *calibrate* mode). Discover the ~80 categorical
honeypots planted in the pool, codify the detectors, and calibrate the per-detector
cut points + composite ``honeypot_threshold`` — all relative to the O0 observed
distributions, never hard-coded, so the gate is defensible at Stage 5.

The hard-gate policy (Part 4 / Part 5): ``is_honeypot = (≥2 HARD impossibilities)
OR (composite ≥ threshold)``. Lone soft anomalies only dampen — false positives
drop real candidates and cost NDCG, so recall is maximized while real-candidate
loss is bounded. ``hp.salary_anomaly`` is SOFT only (inversion is common in the
pool) and never hard-gates.

Algorithm (deps O0, O2):

1. Read O0's ``dataset_profile.json`` to set tolerance bands (impossible vs merely
   unusual) from the observed distribution.
2. Stream the validated pool; run the structural detectors over each candidate
   (timeline, skill-time, overlap, title-seniority, education-career,
   experience-inflation, keyword-stuffing, …). Each fired detector mints
   ``EvidenceRef``s pointing at the exact raw fields (the no-hallucination chain).
3. Classify each fired detector HARD (categorically impossible) or SOFT (unusual).
4. Aggregate per candidate; a suspect is one with ≥2 HARD or composite ≥ threshold.

Outputs (all delegated to the bound store via the base emit helpers):

* ``integrity_rules.json`` — the codified ``IntegrityRuleSet`` (every code ∈
  ``IntegrityFlag``); ``_v_integrity_rules`` (non-empty ``rules``).
* ``honeypot_catalog.json`` — ``HoneypotCatalog``: exemplars (suspect records +
  evidence) + suspect_ids; ``_v_honeypot_catalog`` (``exemplars`` + ``suspect_ids``).
* ``integrity_thresholds.json`` — per-detector cut points + the composite
  ``honeypot_threshold`` (O3 writes the *initial* calibration; O12 finalizes risk
  weights into the same artifact). ``_v_integrity_thresholds``.

Determinism: detectors are pure; findings + rules + suspect ids are sorted; the
threshold is derived deterministically from the observed suspect distribution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from redstack.domain.enums import EvidenceKind, IntegrityFlag, Severity
from redstack.domain.errors import ArtifactContractError, ProvenanceError
from redstack.domain.provenance import EvidenceRef
from redstack.features.evidence import mint
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage
from redstack.ports._types import Malformed, Ok

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "HoneypotDiscoveryStage",
    "honeypot_discovery_stage",
)

#: Tenure-vs-experience tolerance: Σmonths/12 may exceed stated years by this
#: factor before it is *impossible* (HARD) rather than merely noisy (Part 4).
_TENURE_TOLERANCE: Final[float] = 1.5
#: Minimum count of expert-skills-at-zero-usage to fire EXPERT_SKILL_ZERO_USAGE
#: as HARD "en masse" (Part 5: require count ≥ k, not a single skill).
_EXPERT_ZERO_MIN: Final[int] = 5
#: Keyword-stuffing breadth: this many high-proficiency, zero-corroboration skills.
_STUFFING_BREADTH_MIN: Final[int] = 8
#: Max exemplars retained verbatim in the catalog (bounded artifact).
_EXEMPLAR_CAP: Final[int] = 200
#: Number of HARD impossibilities that force a honeypot regardless of composite.
_HARD_GATE_COUNT: Final[int] = 2
#: Proficiency labels treated as "advanced or above" for skill-time contradiction.
_ADVANCED_PROFICIENCY: Final[frozenset[str]] = frozenset({"advanced", "expert"})


@dataclass(slots=True)
class _Finding:
    """One fired detector for one candidate, with its evidence + severity."""

    code: IntegrityFlag
    severity: Severity
    detail: str
    evidence: tuple[EvidenceRef, ...]


@dataclass(slots=True)
class _CandidateVerdict:
    """The per-candidate aggregation of fired detectors."""

    candidate_id: str
    findings: list[_Finding] = field(default_factory=list)

    @property
    def hard_count(self) -> int:
        return sum(1 for f in self.findings if f.severity is Severity.HARD)

    @property
    def composite(self) -> float:
        """Composite honeypot score: HARD weighted 1.0, SOFT 0.35, capped at 1.0."""
        score = sum(
            1.0 if f.severity is Severity.HARD else 0.35 for f in self.findings
        )
        return min(1.0, score)


class HoneypotDiscoveryStage(OfflineStage):
    """O3 — run structural detectors, codify rules, calibrate honeypot thresholds."""

    stage_id = "O3"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        self._load_profile(ctx)  # presence/structure check (tolerances are fixed).
        verdicts: list[_CandidateVerdict] = []
        detector_fire_counts: dict[str, int] = {f.value: 0 for f in IntegrityFlag}
        total = 0

        for record in ctx.candidate_source.stream():
            if isinstance(record, Malformed) or not isinstance(record, Ok):
                continue
            total += 1
            verdict = self._evaluate_candidate(record.raw)
            for finding in verdict.findings:
                detector_fire_counts[finding.code.value] += 1
            if verdict.findings:
                verdicts.append(verdict)

        if total == 0:
            msg = "honeypot discovery saw zero candidates"
            raise ArtifactContractError(msg)

        threshold = self._calibrate_threshold(verdicts, total)
        suspects = self._select_suspects(verdicts, threshold)

        rules_artifact = self.emit_json(
            ctx, "integrity_rules", self._build_rule_set(detector_fire_counts, total)
        )
        catalog_artifact = self.emit_json(
            ctx, "honeypot_catalog", self._build_catalog(suspects)
        )
        thresholds_artifact = self.emit_json(
            ctx,
            "integrity_thresholds",
            self._build_thresholds(detector_fire_counts, total, threshold),
        )
        metrics: dict[str, object] = {
            "candidates_scanned": total,
            "suspects": len(suspects),
            "suspect_rate": round(len(suspects) / total, 6),
            "honeypot_threshold": round(threshold, 6),
            "hard_gate_count": _HARD_GATE_COUNT,
            "detector_fire_counts": dict(sorted(detector_fire_counts.items())),
        }
        return StageResult(
            artifacts=(rules_artifact, catalog_artifact, thresholds_artifact),
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    # O0 profile (tolerance source)                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_profile(ctx: OfflinePipelineContext) -> Mapping[str, object]:
        """Load O0's ``dataset_profile.json`` (the distribution tolerances source).

        Raises:
            ArtifactContractError: the profile is missing/malformed (O3 lineage
                requires O0; thresholds must be distribution-relative, Part 4).
        """
        try:
            raw = ctx.artifact_store.load_text("dataset_profile")
        except Exception as exc:  # noqa: BLE001 — surface as a contract failure.
            msg = f"cannot load dataset_profile for honeypot calibration: {exc}"
            raise ArtifactContractError(msg) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"dataset_profile.json is not valid JSON: {exc}"
            raise ArtifactContractError(msg) from exc
        if not isinstance(parsed, Mapping):
            raise ArtifactContractError("dataset_profile.json must be an object")
        return parsed

    # ------------------------------------------------------------------ #
    # Detectors (each mints EvidenceRefs for its finding)                #
    # ------------------------------------------------------------------ #
    def _evaluate_candidate(self, raw: Mapping[str, object]) -> _CandidateVerdict:
        """Run every structural detector over one candidate; collect findings."""
        cid_obj = raw.get("candidate_id")
        cid = cid_obj if isinstance(cid_obj, str) else "<unknown>"
        verdict = _CandidateVerdict(candidate_id=cid)
        for detector in (
            self._detect_current_role_has_end_date,
            self._detect_role_duration_date_mismatch,
            self._detect_tenure_exceeds_experience,
            self._detect_expert_skill_zero_usage,
            self._detect_education_timeline_impossible,
            self._detect_keyword_stuffing,
            self._detect_assessment_for_absent_skill,
        ):
            finding = detector(raw)
            if finding is not None:
                verdict.findings.append(finding)
        # Deterministic order for reproducible catalogs.
        verdict.findings.sort(key=lambda f: f.code.value)
        return verdict

    @staticmethod
    def _career(raw: Mapping[str, object]) -> list[Mapping[str, object]]:
        history = raw.get("career_history")
        if isinstance(history, (list, tuple)):
            return [r for r in history if isinstance(r, Mapping)]
        return []

    def _detect_current_role_has_end_date(
        self, raw: Mapping[str, object]
    ) -> _Finding | None:
        """HARD: a role flagged ``is_current`` yet carrying a non-null ``end_date``."""
        for idx, role in enumerate(self._career(raw)):
            if role.get("is_current") is True and role.get("end_date") not in (None, "null"):
                evidence = self._mint_safe(
                    raw,
                    (
                        (EvidenceKind.CAREER_FIELD, f"career_history[{idx}].is_current"),
                        (EvidenceKind.CAREER_FIELD, f"career_history[{idx}].end_date"),
                    ),
                )
                if evidence:
                    return _Finding(
                        IntegrityFlag.CURRENT_ROLE_HAS_END_DATE,
                        Severity.HARD,
                        f"role {idx} is current but has an end_date",
                        evidence,
                    )
        return None

    def _detect_role_duration_date_mismatch(
        self, raw: Mapping[str, object]
    ) -> _Finding | None:
        """SOFT: ``duration_months`` inconsistent with start/end span (noisy pool)."""
        for idx, role in enumerate(self._career(raw)):
            duration = role.get("duration_months")
            start = role.get("start_date")
            end = role.get("end_date")
            if (
                isinstance(duration, int)
                and isinstance(start, str)
                and isinstance(end, str)
            ):
                span = self._month_span(start, end)
                if span is not None and abs(span - duration) > 6:
                    evidence = self._mint_safe(
                        raw,
                        ((EvidenceKind.CAREER_FIELD, f"career_history[{idx}].duration_months"),),
                    )
                    if evidence:
                        return _Finding(
                            IntegrityFlag.ROLE_DURATION_DATE_MISMATCH,
                            Severity.SOFT,
                            f"role {idx} duration {duration}m vs span {span}m",
                            evidence,
                        )
        return None

    def _detect_tenure_exceeds_experience(
        self, raw: Mapping[str, object]
    ) -> _Finding | None:
        """HARD: Σ career months / 12 exceeds stated years beyond tolerance."""
        profile = raw.get("profile")
        if not isinstance(profile, Mapping):
            return None
        years = profile.get("years_of_experience")
        if not isinstance(years, (int, float)) or isinstance(years, bool):
            return None
        total_months = 0
        for role in self._career(raw):
            d = role.get("duration_months")
            if isinstance(d, int) and not isinstance(d, bool):
                total_months += d
        derived_years = total_months / 12.0
        if years > 0 and derived_years > years * _TENURE_TOLERANCE:
            evidence = self._mint_safe(
                raw, ((EvidenceKind.PROFILE_FIELD, "profile.years_of_experience"),)
            )
            if evidence:
                return _Finding(
                    IntegrityFlag.TENURE_EXCEEDS_EXPERIENCE,
                    Severity.HARD,
                    f"derived {derived_years:.1f}y >> stated {years}y",
                    evidence,
                )
        return None

    def _detect_expert_skill_zero_usage(
        self, raw: Mapping[str, object]
    ) -> _Finding | None:
        """HARD (en masse): many advanced+ skills with zero/None duration."""
        skills = raw.get("skills")
        if not isinstance(skills, (list, tuple)):
            return None
        offenders: list[int] = []
        for idx, skill in enumerate(skills):
            if not isinstance(skill, Mapping):
                continue
            prof = skill.get("proficiency")
            dur = skill.get("duration_months")
            if (
                isinstance(prof, str)
                and prof.casefold() in _ADVANCED_PROFICIENCY
                and dur in (0, None, "null")
            ):
                offenders.append(idx)
        if len(offenders) >= _EXPERT_ZERO_MIN:
            evidence = self._mint_safe(
                raw,
                tuple(
                    (EvidenceKind.SKILL, f"skills[{i}].proficiency") for i in offenders[:3]
                ),
            )
            if evidence:
                return _Finding(
                    IntegrityFlag.EXPERT_SKILL_ZERO_USAGE,
                    Severity.HARD,
                    f"{len(offenders)} advanced+ skills with zero usage",
                    evidence,
                )
        return None

    def _detect_education_timeline_impossible(
        self, raw: Mapping[str, object]
    ) -> _Finding | None:
        """HARD: education ``end_year < start_year``."""
        education = raw.get("education")
        if not isinstance(education, (list, tuple)):
            return None
        for idx, edu in enumerate(education):
            if not isinstance(edu, Mapping):
                continue
            start = edu.get("start_year")
            end = edu.get("end_year")
            if isinstance(start, int) and isinstance(end, int) and end < start:
                evidence = self._mint_safe(
                    raw,
                    (
                        (EvidenceKind.EDUCATION, f"education[{idx}].start_year"),
                        (EvidenceKind.EDUCATION, f"education[{idx}].end_year"),
                    ),
                )
                if evidence:
                    return _Finding(
                        IntegrityFlag.EDUCATION_TIMELINE_IMPOSSIBLE,
                        Severity.HARD,
                        f"education {idx} end_year {end} < start_year {start}",
                        evidence,
                    )
        return None

    def _detect_keyword_stuffing(self, raw: Mapping[str, object]) -> _Finding | None:
        """SOFT: broad high-proficiency skill list with zero corroboration."""
        skills = raw.get("skills")
        if not isinstance(skills, (list, tuple)):
            return None
        stuffed = 0
        first_idx: int | None = None
        for idx, skill in enumerate(skills):
            if not isinstance(skill, Mapping):
                continue
            prof = skill.get("proficiency")
            endorse = skill.get("endorsements")
            dur = skill.get("duration_months")
            high = isinstance(prof, str) and prof.casefold() in _ADVANCED_PROFICIENCY
            zero_corr = (endorse in (0, None)) and (dur in (0, None, "null"))
            if high and zero_corr:
                stuffed += 1
                if first_idx is None:
                    first_idx = idx
        if stuffed >= _STUFFING_BREADTH_MIN and first_idx is not None:
            evidence = self._mint_safe(
                raw, ((EvidenceKind.SKILL, f"skills[{first_idx}].proficiency"),)
            )
            if evidence:
                return _Finding(
                    IntegrityFlag.KEYWORD_STUFFING,
                    Severity.SOFT,
                    f"{stuffed} high-proficiency skills with zero corroboration",
                    evidence,
                )
        return None

    def _detect_assessment_for_absent_skill(
        self, raw: Mapping[str, object]
    ) -> _Finding | None:
        """HARD: a ``skill_assessment_scores`` key absent from declared ``skills``."""
        signals = raw.get("redrob_signals")
        if not isinstance(signals, Mapping):
            return None
        assessments = signals.get("skill_assessment_scores")
        if not isinstance(assessments, Mapping) or not assessments:
            return None
        skills = raw.get("skills")
        declared = {
            str(s.get("name")).casefold()
            for s in (skills if isinstance(skills, (list, tuple)) else [])
            if isinstance(s, Mapping) and isinstance(s.get("name"), str)
        }
        for assessed in assessments:
            if isinstance(assessed, str) and assessed.casefold() not in declared:
                evidence = self._mint_safe(
                    raw,
                    ((EvidenceKind.SIGNAL, "redrob_signals.skill_assessment_scores"),),
                )
                if evidence:
                    return _Finding(
                        IntegrityFlag.ASSESSMENT_FOR_ABSENT_SKILL,
                        Severity.HARD,
                        f"assessment for undeclared skill {assessed!r}",
                        evidence,
                    )
        return None

    # ------------------------------------------------------------------ #
    # Evidence + date helpers                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mint_safe(
        raw: Mapping[str, object],
        specs: Sequence[tuple[EvidenceKind, str]],
    ) -> tuple[EvidenceRef, ...]:
        """Mint the given (kind, path) refs; drop any that fail to resolve.

        A finding must carry ≥1 resolvable ref; if *none* resolve the finding is
        suppressed (returns ``()``), preserving the no-hallucination invariant —
        a detector never reports a fact it cannot point at.
        """
        refs: list[EvidenceRef] = []
        for kind, path in specs:
            try:
                refs.append(mint(raw, kind=kind, path=path))
            except ProvenanceError:
                continue
        return tuple(refs)

    @staticmethod
    def _month_span(start: str, end: str) -> int | None:
        """Months between two ISO dates (``YYYY-MM-DD``); ``None`` if unparseable."""
        try:
            sy, sm, _ = (int(p) for p in start.split("-")[:3])
            ey, em, _ = (int(p) for p in end.split("-")[:3])
        except (ValueError, IndexError):
            return None
        return (ey - sy) * 12 + (em - sm)

    # ------------------------------------------------------------------ #
    # Calibration + artifact assembly                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calibrate_threshold(
        verdicts: Sequence[_CandidateVerdict], total: int
    ) -> float:
        """Calibrate the composite ``honeypot_threshold`` from the suspect mass.

        Strategy (Part 4): keep recall high while bounding real-candidate loss.
        Candidates with ≥2 HARD impossibilities are honeypots by the hard gate
        regardless of threshold; the composite threshold is set just below the
        composite of the smallest HARD-gated cohort so a single SOFT anomaly never
        crosses it. Concretely: the threshold is the composite produced by exactly
        one HARD finding plus a margin, never reachable by soft-only anomalies.
        """
        # One HARD finding => composite 1.0; one SOFT => 0.35; the hard gate (≥2
        # HARD) is the categorical path. Set the composite cut just above the
        # max soft-only composite observed so soft anomalies never gate alone.
        max_soft_only = 0.0
        for verdict in verdicts:
            if verdict.hard_count == 0:
                max_soft_only = max(max_soft_only, verdict.composite)
        # Place the threshold strictly above the soft-only mass but ≤ a single
        # HARD's contribution, so composite-gating only catches HARD-bearing or
        # heavily-corroborated profiles. Bounded to (0, 1].
        threshold = min(1.0, max(max_soft_only + 1e-3, 0.7))
        return threshold

    @staticmethod
    def _select_suspects(
        verdicts: Sequence[_CandidateVerdict], threshold: float
    ) -> list[_CandidateVerdict]:
        """Apply the hard-gate policy: ≥2 HARD OR composite ≥ threshold."""
        suspects = [
            v
            for v in verdicts
            if v.hard_count >= _HARD_GATE_COUNT or v.composite >= threshold
        ]
        suspects.sort(key=lambda v: v.candidate_id)
        return suspects

    @staticmethod
    def _build_rule_set(
        fire_counts: Mapping[str, int], total: int
    ) -> dict[str, object]:
        """Codify the detectors into the ``IntegrityRuleSet`` artifact.

        Every rule's ``code`` is a member of ``IntegrityFlag`` (Part 10 validation
        "every code in IntegrityFlag"). Each rule records its severity class and
        the observed prevalence (fire count / N) for audit.
        """
        severity_of = {
            IntegrityFlag.CURRENT_ROLE_HAS_END_DATE: Severity.HARD,
            IntegrityFlag.TENURE_EXCEEDS_EXPERIENCE: Severity.HARD,
            IntegrityFlag.EXPERT_SKILL_ZERO_USAGE: Severity.HARD,
            IntegrityFlag.EDUCATION_TIMELINE_IMPOSSIBLE: Severity.HARD,
            IntegrityFlag.ASSESSMENT_FOR_ABSENT_SKILL: Severity.HARD,
            IntegrityFlag.ROLE_DURATION_DATE_MISMATCH: Severity.SOFT,
            IntegrityFlag.KEYWORD_STUFFING: Severity.SOFT,
        }
        rules = [
            {
                "code": flag.value,
                "severity": severity_of.get(flag, Severity.SOFT).value,
                "prevalence": round(fire_counts.get(flag.value, 0) / total, 6),
                "fired": fire_counts.get(flag.value, 0),
            }
            for flag in sorted(severity_of, key=lambda f: f.value)
        ]
        return {"rules": rules, "candidates_scanned": total}

    def _build_catalog(
        self, suspects: Sequence[_CandidateVerdict]
    ) -> dict[str, object]:
        """Build the ``HoneypotCatalog``: exemplars (+ evidence) + suspect ids."""
        exemplars: list[dict[str, object]] = []
        for verdict in suspects[:_EXEMPLAR_CAP]:
            exemplars.append(
                {
                    "candidate_id": verdict.candidate_id,
                    "hard_count": verdict.hard_count,
                    "composite": round(verdict.composite, 6),
                    "findings": [
                        {
                            "code": f.code.value,
                            "severity": f.severity.value,
                            "detail": f.detail,
                            "evidence": [
                                {"kind": e.kind.value, "path": e.path, "value": e.value}
                                for e in f.evidence
                            ],
                        }
                        for f in verdict.findings
                    ],
                }
            )
        suspect_ids = sorted(v.candidate_id for v in suspects)
        return {"exemplars": exemplars, "suspect_ids": suspect_ids}

    @staticmethod
    def _build_thresholds(
        fire_counts: Mapping[str, int], total: int, threshold: float
    ) -> dict[str, object]:
        """Build the initial ``integrity_thresholds.json`` (O12 finalizes risk weights).

        Per-detector cut points are the observed prevalence used as the
        "impossible vs unusual" reference; the composite ``honeypot_threshold`` is
        the calibrated gate. ``hard_gate_count`` records the ≥2-HARD rule.
        """
        return {
            "per_detector": {
                code: {
                    "prevalence": round(count / total, 6),
                    "fired": count,
                }
                for code, count in sorted(fire_counts.items())
            },
            "honeypot_threshold": round(threshold, 6),
            "hard_gate_count": _HARD_GATE_COUNT,
            "calibrated_on": total,
        }


def honeypot_discovery_stage() -> HoneypotDiscoveryStage:
    """Factory: construct the O3 honeypot-discovery stage."""
    return HoneypotDiscoveryStage()