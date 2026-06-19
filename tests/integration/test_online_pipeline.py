

from __future__ import annotations

import csv
import io
import itertools
import math
import random
import re
import statistics
from datetime import date
from typing import Final

import numpy as np

from redstack.domain.enums import BuildStage, EligibilityCode, EvidenceKind
from redstack.domain.errors import RepresentationStageError
from redstack.domain.ids import CandidateId
from redstack.pipelines.online import stages
from redstack.pipelines.online.pipeline import (
    OnlinePipeline,
    OnlineRunConfig,
    OnlineRunContext,
)
from tests.fixtures.fake_ports import (
    CapturingRunReportSink,
    CapturingSubmissionSink,
    FixedEntropy,
    InMemoryArtifactStore,
    InMemoryVectorStore,
    ListCandidateSource,
    StubEmbeddingModel,
)

DIM: Final[int] = 16
AS_OF: Final[date] = date(2024, 6, 18)
SEED: Final[int] = 1
COMPETENCY_GROUPS: Final[tuple[str, ...]] = (
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
_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^CAND_[0-9]{7}$")

_TITLES = (
    "Senior ML Engineer",
    "ML Engineer",
    "Staff ML Engineer",
    "Applied Scientist",
    "Data Scientist",
    "Software Engineer",
)
_COMPANIES = (
    "Acme Tech",
    "Beta Labs",
    "Gamma Systems",
    "Delta Software",
    "Epsilon AI",
    "Zeta Cloud",
)
_INDUSTRIES = ("software", "saas", "internet", "fintech", "ecommerce")
_CITIES = ("Pune", "Bengaluru", "Hyderabad", "Mumbai", "Noida", "Chennai", "Remote-US")
_SKILL_POOL = (
    "python",
    "retrieval",
    "ranking",
    "recommendation",
    "pytorch",
    "tensorflow",
    "search",
    "nlp",
    "transformers",
    "kubernetes",
    "sql",
    "spark",
)
_DESCRIPTIONS = (
    "Built retrieval and ranking models, deployed to production, owned the roadmap.",
    "Implemented recommendation systems and shipped production features.",
    "Led a small team building search infrastructure at scale, on-call rotations.",
    "Researched novel approaches; published a paper; mostly prototype-only work.",
)


# --------------------------------------------------------------------------- #
# Fixture builders: a diverse synthetic candidate pool + two hand-crafted     #
# exemplars (a honeypot, a consulting-only hard-block) per Testing Strategy   #
# §10/§16's fixture-family requirements.                                     #
# --------------------------------------------------------------------------- #
def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def _make_candidate(i: int, rng_: random.Random) -> dict[str, object]:
    cid = f"CAND_{i:07d}"
    years = round(rng_.uniform(2.0, 12.0), 1)
    title = rng_.choice(_TITLES)
    company = rng_.choice(_COMPANIES)
    industry = rng_.choice(_INDUSTRIES)
    city = rng_.choice(_CITIES)
    country = "India" if city != "Remote-US" else "USA"
    n_skills = rng_.randint(2, 6)
    skills = [
        {
            "name": name,
            "proficiency": rng_.choice(
                ["beginner", "intermediate", "advanced", "expert"]
            ),
            "endorsements": rng_.randint(0, 30),
            "duration_months": rng_.choice([None, 0, rng_.randint(1, 60)]),
        }
        for name in rng_.sample(_SKILL_POOL, n_skills)
    ]
    n_positions = rng_.randint(1, 3)
    career_history = []
    cursor_year = 2024
    for p in range(n_positions):
        start_year = cursor_year - rng_.randint(1, 4)
        career_history.append(
            {
                "company": rng_.choice(_COMPANIES),
                "title": rng_.choice(_TITLES),
                "start_date": f"{start_year}-01-01",
                "end_date": None if p == 0 else f"{cursor_year}-01-01",
                "duration_months": (cursor_year - start_year) * 12,
                "is_current": p == 0,
                "industry": rng_.choice(_INDUSTRIES),
                "company_size": rng_.choice(
                    ["1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000"]
                ),
                "description": rng_.choice(_DESCRIPTIONS),
            }
        )
        cursor_year = start_year
    assessment_scores = {
        s["name"]: round(rng_.uniform(40, 99), 1) for s in skills if rng_.random() < 0.5
    }
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": f"Candidate {i}",
            "headline": f"{title} at {company}",
            "summary": rng_.choice(_DESCRIPTIONS),
            "location": city,
            "country": country,
            "years_of_experience": years,
            "current_title": title,
            "current_company": company,
            "current_company_size": "201-500",
            "current_industry": industry,
        },
        "career_history": career_history,
        "education": [
            {
                "institution": rng_.choice(
                    ["IIT Bombay", "BITS Pilani", "Anna University"]
                ),
                "degree": rng_.choice(["BTech", "MTech", "BSc"]),
                "field_of_study": rng_.choice(["Computer Science", "Mathematics"]),
                "start_year": 2010 + rng_.randint(0, 8),
                "end_year": 2014 + rng_.randint(0, 8),
                "grade": None,
                "tier": rng_.choice(["tier_1", "tier_2", "tier_3", "unknown"]),
            }
        ],
        "skills": skills,
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": round(rng_.uniform(40, 100), 1),
            "signup_date": "2019-01-01",
            "last_active_date": rng_.choice(["2024-06-01", "2023-01-01", "2024-05-15"]),
            "open_to_work_flag": rng_.choice([True, False]),
            "profile_views_received_30d": rng_.randint(0, 50),
            "applications_submitted_30d": rng_.randint(0, 10),
            "recruiter_response_rate": round(rng_.uniform(0, 1), 2),
            "avg_response_time_hours": round(rng_.uniform(0.5, 48), 1),
            "skill_assessment_scores": assessment_scores,
            "connection_count": rng_.randint(0, 1000),
            "endorsements_received": rng_.randint(0, 100),
            "notice_period_days": rng_.choice([0, 15, 30, 45, 60, 90, 120]),
            "expected_salary_range_inr_lpa": {
                "min": rng_.uniform(10, 50),
                "max": rng_.uniform(50, 80),
            },
            "preferred_work_mode": rng_.choice(
                ["remote", "hybrid", "onsite", "flexible"]
            ),
            "willing_to_relocate": rng_.choice([True, False]),
            "github_activity_score": rng_.choice(
                [-1.0, round(rng_.uniform(0, 100), 1)]
            ),
            "search_appearance_30d": rng_.randint(0, 20),
            "saved_by_recruiters_30d": rng_.randint(0, 10),
            "interview_completion_rate": round(rng_.uniform(0, 1), 2),
            "offer_acceptance_rate": rng_.choice([-1.0, round(rng_.uniform(0, 1), 2)]),
            "verified_email": rng_.choice([True, False]),
            "verified_phone": rng_.choice([True, False]),
            "linkedin_connected": rng_.choice([True, False]),
        },
    }


def _make_honeypot(cid: str) -> dict[str, object]:
    """Two corroborating HARD impossibilities: tenure >> stated experience,
    and >=5 ``EXPERT``-proficiency skills claimed at zero duration/usage --
    R0's locked ``expert_zero_usage_min_count`` (online ``stages.r0_load``)."""
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Honeypot Candidate",
            "headline": "ML Engineer",
            "summary": "Honeypot exemplar.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 2.0,
            "current_title": "ML Engineer",
            "current_company": "Acme Tech",
            "current_company_size": "201-500",
            "current_industry": "software",
        },
        "career_history": [
            {
                "company": "Acme Tech",
                "title": "ML Engineer",
                "start_date": "2014-01-01",
                "end_date": None,
                "duration_months": 120,
                "is_current": True,
                "industry": "software",
                "company_size": "201-500",
                "description": "Built ML systems.",
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "BTech",
                "field_of_study": "Computer Science",
                "start_year": 2010,
                "end_year": 2014,
                "grade": None,
                "tier": "tier_1",
            }
        ],
        "skills": [
            {
                "name": "python",
                "proficiency": "expert",
                "endorsements": 0,
                "duration_months": 0,
            },
            {
                "name": "pytorch",
                "proficiency": "expert",
                "endorsements": 0,
                "duration_months": None,
            },
            {
                "name": "search",
                "proficiency": "expert",
                "endorsements": 0,
                "duration_months": 0,
            },
            {
                "name": "tensorflow",
                "proficiency": "expert",
                "endorsements": 0,
                "duration_months": 0,
            },
            {
                "name": "ranking",
                "proficiency": "expert",
                "endorsements": 0,
                "duration_months": None,
            },
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 50.0,
            "signup_date": "2023-01-01",
            "last_active_date": "2024-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 1,
            "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 10.0,
            "skill_assessment_scores": {},
            "connection_count": 10,
            "endorsements_received": 0,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 30.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": -1.0,
            "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0,
            "interview_completion_rate": 0.5,
            "offer_acceptance_rate": -1.0,
            "verified_email": True,
            "verified_phone": False,
            "linkedin_connected": False,
        },
    }


def _make_consulting_only(cid: str) -> dict[str, object]:
    """An entire career at consulting firms -> CONSULTING_FIRMS_ONLY_CAREER (HARD)."""
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Consulting Candidate",
            "headline": "Consultant",
            "summary": "Consulting engagements across clients.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Senior Consultant",
            "current_company": "TCS",
            "current_company_size": "10001+",
            "current_industry": "consulting",
        },
        "career_history": [
            {
                "company": "TCS",
                "title": "Senior Consultant",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 48,
                "is_current": True,
                "industry": "consulting",
                "company_size": "10001+",
                "description": (
                    "Consulting engagement delivering client-facing staff "
                    "augmentation services."
                ),
            },
            {
                "company": "Infosys",
                "title": "Consultant",
                "start_date": "2018-01-01",
                "end_date": "2019-12-31",
                "duration_months": 24,
                "is_current": False,
                "industry": "consulting",
                "company_size": "10001+",
                "description": "Client-facing consulting delivery, staff augmentation.",
            },
        ],
        "education": [
            {
                "institution": "Anna University",
                "degree": "BTech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": None,
                "tier": "tier_2",
            }
        ],
        "skills": [
            {
                "name": "python",
                "proficiency": "intermediate",
                "endorsements": 5,
                "duration_months": 24,
            },
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 70.0,
            "signup_date": "2020-01-01",
            "last_active_date": "2024-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 5,
            "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.6,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {},
            "connection_count": 200,
            "endorsements_received": 5,
            "notice_period_days": 60,
            "expected_salary_range_inr_lpa": {"min": 15.0, "max": 25.0},
            "preferred_work_mode": "onsite",
            "willing_to_relocate": False,
            "github_activity_score": -1.0,
            "search_appearance_30d": 1,
            "saved_by_recruiters_30d": 0,
            "interview_completion_rate": 0.6,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


def _diverse_pool(n: int, *, seed: int = SEED) -> list[dict[str, object]]:
    rng = random.Random(seed)
    return [_make_candidate(i, rng) for i in range(1, n + 1)]


HONEYPOT_IDS: Final[tuple[str, ...]] = ("CAND_9000001", "CAND_9000002", "CAND_9000003")
CONSULTING_IDS: Final[tuple[str, ...]] = ("CAND_9000004", "CAND_9000005")


def _full_pool() -> list[dict[str, object]]:
    pool = _diverse_pool(130)
    pool.extend(_make_honeypot(cid) for cid in HONEYPOT_IDS)
    pool.extend(_make_consulting_only(cid) for cid in CONSULTING_IDS)
    return pool


# --------------------------------------------------------------------------- #
# Artifact / port assembly.                                                   #
# --------------------------------------------------------------------------- #
def _build_artifact_store(*, dim: int = DIM) -> InMemoryArtifactStore:
    """Build the artifact set ``r0_load`` actually reads (Online Part 1/§R0).

    Keys/shapes mirror ``pipelines/online/stages.py``'s real reads exactly
    (``scoring_weights`` is YAML-parsed text; ``jd_concepts``/``anchor_vectors``
    and ``archetypes``/``centroids`` are split pairs aligned by sorted-id /
    catalog-key order; ``integrity_rules`` -- not ``integrity_thresholds`` --
    is where per-flag severity now lives) rather than the offline registry's
    on-disk relative paths, which are not artifact keys.
    """
    rng = random.Random(7)
    anchor_polarity = {f"jd.{group}": "positive" for group in COMPETENCY_GROUPS}
    anchor_polarity["jd.neg1"] = "negative"
    ordered_anchor_ids = sorted(anchor_polarity)
    anchor_vectors = np.stack(
        [_unit([rng.gauss(0, 1) for _ in range(dim)]) for _ in ordered_anchor_ids]
    ).astype(np.float32)
    jd_concepts = {
        "anchors": [
            {"id": aid, "polarity": anchor_polarity[aid], "text": aid}
            for aid in ordered_anchor_ids
        ]
    }
    archetype_catalog = {
        "0": {"archetype_id": 0, "label": "other", "is_target_archetype": False},
        "1": {"archetype_id": 1, "label": "ideal-fit", "is_target_archetype": True},
    }
    centroids = np.stack(
        [_unit([rng.gauss(0, 1) for _ in range(dim)]) for _ in range(2)]
    ).astype(np.float32)

    return InMemoryArtifactStore.build(
        {
            "scoring_weights": (
                "json",
                {
                    "layout_version": "1.1.0",
                    "weights": {
                        "skill_match": 0.25,
                        "semantic_fit": 0.2,
                        "career_fit": 0.15,
                        "experience_fit": 0.15,
                        "education_fit": 0.1,
                        "credibility": 0.1,
                        "archetype_fit": 0.05,
                    },
                    "neutral_prior": 0.5,
                },
            ),
            "integrity_thresholds": (
                "json",
                {
                    "honeypot_threshold": 0.7,
                    "risk_weights": {
                        "tenure_exceeds_experience": 0.4,
                        "role_duration_date_mismatch": 0.3,
                        "current_role_has_end_date": 0.3,
                        "expert_skill_zero_usage": 0.4,
                        "education_timeline_impossible": 0.4,
                        "experience_predates_plausible_start": 0.4,
                        "assessment_for_absent_skill": 0.3,
                    },
                },
            ),
            "integrity_rules": (
                "json",
                {
                    "rules": [
                        {"code": code, "severity": "hard"}
                        for code in (
                            "tenure_exceeds_experience",
                            "role_duration_date_mismatch",
                            "current_role_has_end_date",
                            "expert_skill_zero_usage",
                            "education_timeline_impossible",
                            "experience_predates_plausible_start",
                            "assessment_for_absent_skill",
                            "keyword_stuffing",
                        )
                    ]
                },
            ),
            "eligibility_rules": ("yaml", "hard_blocks: []\nsoft_penalties: []\n"),
            "behavioral_weights": (
                "json",
                {"bounds": {"lower": 0.5, "upper": 1.25}, "neutral_multiplier": 1.0},
            ),
            "jd_concepts": ("json", jd_concepts),
            "anchor_vectors": ("npy", anchor_vectors),
            "archetypes": ("json", {"archetypes": archetype_catalog}),
            "centroids": ("npy", centroids),
        },
        layout_version="1.1.0",
        embedding_model_id="stub-embedder-v1",
        embedding_dim=dim,
    )


def _build_vector_store(
    candidate_ids: list[str], *, dim: int = DIM, missing: frozenset[str] = frozenset()
) -> InMemoryVectorStore:
    rng = random.Random(11)
    vectors = {
        CandidateId(cid): _unit([rng.gauss(0, 1) for _ in range(dim)])
        for cid in candidate_ids
        if cid not in missing
    }
    return InMemoryVectorStore(dim, vectors)


def _build_pipeline(
    candidates: list[dict[str, object]],
    *,
    missing_vectors: frozenset[str] = frozenset(),
) -> tuple[OnlinePipeline, CapturingSubmissionSink, CapturingRunReportSink]:
    candidate_ids = [str(c["candidate_id"]) for c in candidates]
    artifact_store = _build_artifact_store()
    vector_store = _build_vector_store(candidate_ids, missing=missing_vectors)
    embedding_model = StubEmbeddingModel(dim=DIM)
    source = ListCandidateSource(candidates)
    submission_sink = CapturingSubmissionSink()
    run_report_sink = CapturingRunReportSink()
    entropy = FixedEntropy.online(SEED, AS_OF)
    config = OnlineRunConfig(
        participant_id="test_participant",
        code_version="test",
        config_hash="test",
        seed=SEED,
    )
    pipeline = OnlinePipeline(
        config=config,
        candidate_source=source,
        vector_store=vector_store,
        embedding_model=embedding_model,
        submission_sink=submission_sink,
        run_report_sink=run_report_sink,
        entropy=entropy,
        artifact_store=artifact_store,
    )
    return pipeline, submission_sink, run_report_sink


def _build_context(
    candidates: list[dict[str, object]],
    *,
    missing_vectors: frozenset[str] = frozenset(),
) -> OnlineRunContext:
    """Build an ``OnlineRunContext`` directly (bypassing ``OnlinePipeline.run``)
    for tests that exercise individual R-stages rather than the full run."""
    pipeline, _, _ = _build_pipeline(candidates, missing_vectors=missing_vectors)
    config = pipeline._config
    loaded = stages.r0_load(
        artifact_store=pipeline._artifact_store,
        embedding_model=pipeline._embedding_model,
        vector_store=pipeline._vector_store,
        entropy=pipeline._entropy,
        config=config,
    )
    return OnlineRunContext(
        config=config,
        candidate_source=pipeline._candidate_source,
        vector_store=pipeline._vector_store,
        embedding_model=pipeline._embedding_model,
        submission_sink=pipeline._submission_sink,
        run_report_sink=pipeline._run_report_sink,
        entropy=pipeline._entropy,
        artifact_store=pipeline._artifact_store,
        artifacts=loaded.artifacts,
        manifest=loaded.manifest,
        as_of=pipeline._entropy.as_of(),
        seed=pipeline._entropy.seed,
        input_file_sha256="f" * 64,
    )


def _parse_csv_rows(csv_text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(csv_text)))[1:]


# =========================================================================== #
# Tests                                                                       #
# =========================================================================== #
def test_full_run_produces_validator_compliant_submission() -> None:
    """The organizer's structural rules: header, 100 rows, ranks 1..100,
    non-increasing scores, unique valid ids, id-ascending tie-break."""
    pipeline, submission_sink, _ = _build_pipeline(_full_pool())
    result = pipeline.run(input_file_sha256="d" * 64)

    assert result.row_count == 100
    assert result.within_budget

    csv_text = submission_sink.last_csv_text
    assert csv_text is not None
    lines = csv_text.splitlines()
    assert lines[0] == "candidate_id,rank,score,reasoning"
    assert len(lines) == 101

    rows = _parse_csv_rows(csv_text)
    assert len(rows) == 100

    ranks = [int(row[1]) for row in rows]
    assert ranks == list(range(1, 101))

    ids = [row[0] for row in rows]
    assert len(set(ids)) == 100
    assert all(_ID_PATTERN.match(cid) for cid in ids)

    scores = [float(row[2]) for row in rows]
    for earlier, later in itertools.pairwise(scores):
        assert later <= earlier, "score must be non-increasing by rank"

    for earlier_row, later_row in itertools.pairwise(rows):
        if float(earlier_row[2]) == float(later_row[2]):
            assert earlier_row[0] < later_row[0], "tie must break by ascending id"


def test_build_stage_progression_through_r4() -> None:
    """A representation advances PARSED->FEATURED->SITUATED->GATED exactly,
    and re-attaching an already-populated slice raises."""
    pool = _full_pool()
    ctx = _build_context(pool)

    ingested = stages.r1_ingest(ctx)
    assert all(c.representation.stage is BuildStage.PARSED for c in ingested)

    featured = stages.r2_features(ctx, ingested)
    assert all(rep.stage is BuildStage.FEATURED for rep in featured.representations)
    assert featured.cqv.shape[0] == len(ingested)

    situated = stages.r3_semantic(ctx, featured)
    assert all(rep.stage is BuildStage.SITUATED for rep in situated.representations)

    gated = stages.r4_gates(ctx, situated)
    assert all(rep.stage is BuildStage.GATED for rep in gated.representations)
    assert gated.floor_mask.shape[0] == len(ingested)

    # Re-attaching features at GATED (skipping back) must raise, not silently mutate.
    sample_gated_rep = gated.representations[0]
    sample_featured_rep = featured.representations[0]
    try:
        sample_gated_rep.with_features(
            career=sample_featured_rep.require_career(),
            credibility=sample_featured_rep.require_credibility(),
            behavioral=sample_featured_rep.require_behavioral(),
            logistics=sample_featured_rep.require_logistics(),
        )
    except RepresentationStageError:
        pass
    else:
        raise AssertionError("expected RepresentationStageError on stage regression")


def test_floor_mask_excludes_known_honeypots_and_consulting_only() -> None:
    """Hand-crafted honeypot + consulting-only exemplars never reach the top-100,
    and the top-100 honeypot rate stays well under the 10% disqualification gate."""
    pipeline, _, _ = _build_pipeline(_full_pool())
    result = pipeline.run(input_file_sha256="e" * 64)

    assert result.honeypot_rate_top100 < 0.10

    ctx = _build_context(_full_pool())
    ingested = stages.r1_ingest(ctx)
    featured = stages.r2_features(ctx, ingested)
    situated = stages.r3_semantic(ctx, featured)
    gated = stages.r4_gates(ctx, situated)

    by_id = {rep.candidate_id: rep for rep in gated.representations}
    for honeypot_id in HONEYPOT_IDS:
        integrity = by_id[CandidateId(honeypot_id)].integrity
        assert integrity is not None
        assert integrity.is_honeypot, f"{honeypot_id} should be detected as a honeypot"
    for consulting_id in CONSULTING_IDS:
        eligibility = by_id[CandidateId(consulting_id)].eligibility
        assert eligibility is not None
        assert not eligibility.is_eligible, f"{consulting_id} should fail eligibility"

    scored = stages.r5_score(ctx, gated)
    ranking = stages.r6_rank(ctx, scored)
    ranked_ids = {str(r.candidate_id) for r in ranking.ordered}
    for bad_id in (*HONEYPOT_IDS, *CONSULTING_IDS):
        assert bad_id not in ranked_ids, f"{bad_id} must not occupy a top-100 rank"


def test_reasoning_is_evidence_grounded_and_jd_linked() -> None:
    """Every top-100 clause cites real evidence; every reasoning links to the JD."""
    ctx = _build_context(_full_pool())
    ingested = stages.r1_ingest(ctx)
    featured = stages.r2_features(ctx, ingested)
    situated = stages.r3_semantic(ctx, featured)
    gated = stages.r4_gates(ctx, situated)
    scored = stages.r5_score(ctx, gated)
    ranking = stages.r6_rank(ctx, scored)
    reasoned = stages.r7_reason(ctx, ranking, gated)

    raw_by_id = {c.candidate_id: c.raw for c in ingested}
    for ranked in reasoned.ordered:
        reasoning = ranked.reasoning
        assert reasoning is not None
        assert reasoning.rendered.strip()
        assert any(clause.jd_link is not None for clause in reasoning.clauses)
        raw = raw_by_id[ranked.candidate_id]
        raw_dump = raw.model_dump(mode="json")
        for clause in reasoning.clauses:
            assert clause.evidence, "every clause must carry >=1 EvidenceRef"
            for ref in clause.evidence:
                if ref.kind is EvidenceKind.DERIVED:
                    # A DERIVED ref cites a computed fact (its own minted
                    # value is the source of truth), not a literal
                    # RawCandidate path -- nothing to resolve against raw_dump.
                    continue
                _assert_resolves(raw_dump, ref.path)


def _assert_resolves(raw_dump: dict[str, object], path: str) -> None:
    """Walk a dotted/bracket evidence path and assert it resolves in ``raw_dump``."""
    current: object = raw_dump
    for segment in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if segment.startswith("["):
            assert isinstance(current, list)
            index = int(segment[1:-1])
            assert 0 <= index < len(current)
            current = current[index]
        else:
            assert isinstance(current, dict), f"path {path!r} does not resolve"
            assert segment in current, f"field {segment!r} in path {path!r} not found"
            current = current[segment]


def test_reasoning_has_no_identical_or_templated_text() -> None:
    """Stage-4 variation: no two top-100 reasonings render identically."""
    pipeline, submission_sink, _ = _build_pipeline(_full_pool())
    pipeline.run(input_file_sha256="a" * 64)
    csv_text = submission_sink.last_csv_text
    assert csv_text is not None
    rows = _parse_csv_rows(csv_text)
    rendered = [row[3] for row in rows]
    assert len(set(rendered)) == len(rendered), "all top-100 reasonings must differ"


def test_cold_start_semantic_miss_falls_back_to_encoder() -> None:
    """A candidate id absent from the vector store still gets a semantic profile
    via the onnx-fallback encoder path (R3 never fails on a miss)."""
    pool = _full_pool()
    miss_id = str(pool[0]["candidate_id"])
    ctx = _build_context(pool, missing_vectors=frozenset({miss_id}))
    ingested = stages.r1_ingest(ctx)
    featured = stages.r2_features(ctx, ingested)
    situated = stages.r3_semantic(ctx, featured)
    rep = next(r for r in situated.representations if str(r.candidate_id) == miss_id)
    assert rep.semantic is not None
    assert math.isfinite(float(rep.semantic.net_semantic_fit))
    assert -1.0 <= float(rep.semantic.positive_fit) <= 1.0


def test_structurally_disqualified_skip_embedding_and_never_hit_fallback() -> None:
    """Consulting-only candidates settle CONSULTING_FIRMS_ONLY_CAREER at R2 from
    career/credibility alone, before any vector lookup -- even with their vector
    missing from the store (which would otherwise force the onnx fallback), R3
    must never ask the embedder about them, and R4 must still floor them."""
    pool = _full_pool()
    ctx = _build_context(pool, missing_vectors=frozenset(CONSULTING_IDS))
    embedding_model = ctx.embedding_model
    assert isinstance(embedding_model, StubEmbeddingModel)

    ingested = stages.r1_ingest(ctx)
    featured = stages.r2_features(ctx, ingested)
    for cid in CONSULTING_IDS:
        findings = featured.structural_findings.get(CandidateId(cid))
        assert findings is not None
        assert any(
            f.code is EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER for f in findings
        )

    situated = stages.r3_semantic(ctx, featured)
    # The only missing vectors belong to the consulting-only candidates; if R3
    # had not skipped them, this miss would force exactly this fallback call.
    assert embedding_model.encoded_texts == []

    gated = stages.r4_gates(ctx, situated)
    by_id = {rep.candidate_id: rep for rep in gated.representations}
    for cid in CONSULTING_IDS:
        eligibility = by_id[CandidateId(cid)].eligibility
        assert eligibility is not None
        assert not eligibility.is_eligible
        assert eligibility.hard_blocks
        assert eligibility.soft_penalties == ()


def test_determinism_same_input_runs_byte_identical() -> None:
    """Two runs on identical inputs produce byte-identical CSV + reproducible block."""
    pool = _full_pool()

    pipeline1, sink1, report_sink1 = _build_pipeline(pool)
    result1 = pipeline1.run(input_file_sha256="c" * 64)

    pipeline2, sink2, report_sink2 = _build_pipeline(pool)
    result2 = pipeline2.run(input_file_sha256="c" * 64)

    assert sink1.last_csv_text == sink2.last_csv_text
    assert result1.output_sha256 == result2.output_sha256
    report1 = report_sink1.last_report
    report2 = report_sink2.last_report
    assert report1 is not None
    assert report2 is not None
    assert report1["reproducible"] == report2["reproducible"]


def test_run_report_structure_and_budget() -> None:
    """The run report's reproducible block carries every spec-required field."""
    pipeline, _, run_report_sink = _build_pipeline(_full_pool())
    result = pipeline.run(input_file_sha256="9" * 64)

    report = run_report_sink.last_report
    assert report is not None
    reproducible = report["reproducible"]
    assert isinstance(reproducible, dict)
    for key in (
        "code_version",
        "config_hash",
        "manifest_hash",
        "artifact_hashes",
        "input_file_sha256",
        "candidate_count",
        "output_sha256",
        "honeypot_count_top100",
        "honeypot_rate",
        "eligibility_summary",
        "score_distribution_digest",
    ):
        assert key in reproducible, f"missing reproducible field: {key}"

    assert reproducible["candidate_count"] == len(_full_pool())
    budget = report["budget"]
    assert isinstance(budget, dict)
    assert budget["within_budget"] is True
    assert result.within_budget
    assert statistics.mean(result.stage_timings_ms.values()) >= 0.0
