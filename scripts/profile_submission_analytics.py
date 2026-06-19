#!/usr/bin/env python3
"""High-fidelity candidate triage & validation suite for ``submission.csv``.

Supersedes the old ``debug_top_candidates.py``, which scored candidates with
its own ad-hoc keyword set and threshold heuristics living nowhere near the
real pipeline -- exactly the kind of duplicated-logic drift that produced
several of the bugs fixed earlier in this audit (a stale ``_PRODUCT_INDUSTRIES``
list, a flat-mean ``skill_match``, a leaky lexicon). This script carries zero
parallel reimplementation of that math.

Instead it re-runs the *real* R0-R7 online stages
(``redstack.pipelines.online.stages``) restricted to just the candidates in
``artifacts/submission.csv`` -- not the full 100k pool. Every per-candidate
computation in R2-R5 is independent of every other candidate (no population-
relative normalization anywhere in the gate/scoring path), so re-running the
real stages on this 100-row slice reproduces byte-identical results to the
full run, just fast: ``candidate_vectors.parquet`` already holds these
candidates' embeddings from the original O-pipeline build, so R3 is a pure
lookup against the precomputed store -- no live ONNX inference needed.

Every number this dashboard prints -- skill_match's top-3-of-9 selection, the
in_career/trust/semantic fusion, the eligibility hard-block/soft-penalty
codes, the scoring weights -- is read directly off the live
``EligibilityEngine``/``ScoringEngine``/``features.skills`` objects. Re-run
this script any time those change; there is nothing here to keep in sync by
hand.

This is a development tool outside ``src/redstack/pipelines/online`` and
``src/redstack/engines``, so unlike those packages it may import adapters
directly (CLAUDE.md's Online Containment Rule binds the online *runtime*
package, not ad-hoc analysis scripts).

Usage::

    uv run python scripts/profile_submission_analytics.py

Reads ``artifacts/submission.csv`` + ``data/raw/candidates.jsonl`` +
``artifacts/run_report.json``; writes ``artifacts/debug_top100_lean.json``
and ``artifacts/debug_dashboard.md``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from redstack.adapters.artifact_store_fs import FilesystemArtifactStoreAdapter
from redstack.adapters.candidate_jsonl import JsonlCandidateSourceAdapter
from redstack.adapters.entropy import OnlineEntropy
from redstack.adapters.onnx_embedder import OnnxEmbeddingModelAdapter
from redstack.adapters.run_report_json import JsonRunReportSinkAdapter
from redstack.adapters.submission_csv import CsvSubmissionSinkAdapter
from redstack.adapters.vector_store_parquet import ParquetSemanticVectorStoreAdapter
from redstack.config.determinism import apply_determinism, build_onnx_session_options
from redstack.config.loader import load_config
from redstack.config.schema import RunMode
from redstack.domain.candidate.eligibility import EligibilityReport
from redstack.domain.ids import CandidateId
from redstack.domain.scoring import ScoredCandidate
from redstack.features.extraction import _LEXICON_TOKENS
from redstack.features.skills import _tokenize
from redstack.pipelines.context import canonical_config_hash
from redstack.pipelines.online import stages
from redstack.pipelines.online.pipeline import OnlineRunConfig, OnlineRunContext
from redstack.ports._types import ArtifactKey

# --------------------------------------------------------------------------- #
# Paths.                                                                      #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_PATH = REPO_ROOT / "artifacts" / "submission.csv"
DATASET_PATH = REPO_ROOT / "data" / "raw" / "candidates.jsonl"
RUN_REPORT_PATH = REPO_ROOT / "artifacts" / "run_report.json"
CONFIGS_ROOT = REPO_ROOT / "configs"
LEAN_JSON_PATH = REPO_ROOT / "artifacts" / "debug_top100_lean.json"
DASHBOARD_PATH = REPO_ROOT / "artifacts" / "debug_dashboard.md"

# Real pipeline constants, imported (not copied) so this script can never
# drift from the live skill_match / lexicon definitions.
COMPETENCY_GROUPS: tuple[str, ...] = stages._COMPETENCY_GROUPS
SKILL_MATCH_TOP_K: int = stages._SKILL_MATCH_TOP_K
LEXICON_TOKENS: dict[str, frozenset[str]] = dict(_LEXICON_TOKENS)

# Dabbler-pattern threshold: a competency group counts as "claimed but never
# evidenced on the job" when its in_career cell is exactly zero (no lexicon
# token from the group appears anywhere in the candidate's own career
# descriptions) while its trust cell -- driven purely by self-reported
# endorsements/duration/assessment, none of it career-text-anchored -- clears
# the same 0.5 "credible" bar EligibilityEngine/CredibilityProfile use
# elsewhere (features/extraction.py's ``is_credible = trust_value >= 0.5``).
DABBLER_TRUST_FLOOR = 0.5


# --------------------------------------------------------------------------- #
# Step 1 -- read the submission, pull just those raw records out of the pool. #
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_submission_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        sys.exit(f"[-] submission missing at {path} -- run `make rank` first")
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _extract_target_lines(dataset_path: Path, target_ids: set[str]) -> dict[str, str]:
    """One streaming pass over the full pool; keep only the submitted lines."""
    if not dataset_path.is_file():
        sys.exit(f"[-] candidate pool missing at {dataset_path}")
    found: dict[str, str] = {}
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or not any(tid in line for tid in target_ids):
                continue
            record = json.loads(line)
            cid = record.get("candidate_id")
            if cid in target_ids and cid not in found:
                found[cid] = line
                if len(found) == len(target_ids):
                    break
    missing = target_ids - found.keys()
    if missing:
        sys.exit(
            f"[-] submission references candidates absent from the pool: {missing}"
        )
    return found


# --------------------------------------------------------------------------- #
# Step 2 -- bind the exact adapters the real CLI uses, run R0-R7 for real.   #
# --------------------------------------------------------------------------- #
def _build_context(input_path: Path) -> OnlineRunContext:
    """Mirror ``pipelines.online.compose.run_online_rank``'s adapter wiring.

    Stops short of ``OnlinePipeline.run()`` so the caller can drive R0-R7
    directly and keep the rich intermediate objects (``GatedSet``,
    ``ScoredSet``) instead of just the terminal CSV row.
    """
    resolved = load_config(CONFIGS_ROOT, RunMode.ONLINE, None)
    apply_determinism(resolved.determinism)
    online = resolved.online
    if online is None:
        sys.exit("[-] loaded config has no 'online' runtime block")

    artifacts_root = Path(resolved.paths.artifacts_root).resolve()
    artifact_store = FilesystemArtifactStoreAdapter(artifacts_root)
    manifest = artifact_store.manifest()

    run_config = OnlineRunConfig(
        participant_id="debug-analytics",
        code_version="debug-analytics",
        config_hash=canonical_config_hash(resolved),
        seed=resolved.determinism.seed,
        floor_sentinel=online.score_presentation.floor_sentinel,
        ranking_size=online.top_k,
        score_decimals=online.score_presentation.decimals,
        abort_on_malformed=online.malformed_record_policy.value == "abort",
        budget_limit_seconds=resolved.budget.max_wall_seconds,
        blas_threads=resolved.determinism.omp_num_threads,
        onnx_intra_op_threads=resolved.determinism.onnx_intra_op_threads,
        onnx_inter_op_threads=resolved.determinism.onnx_inter_op_threads,
    )
    candidate_source = JsonlCandidateSourceAdapter(input_path)
    vector_store = ParquetSemanticVectorStoreAdapter(
        artifact_store,
        vectors_key=ArtifactKey("candidate_vectors"),
        expected_dim=manifest.embedding.dim,
    )
    session_options = build_onnx_session_options(resolved.determinism)
    embedding_model = OnnxEmbeddingModelAdapter(
        artifact_store,
        model_key=ArtifactKey("encoder"),
        tokenizer_key=ArtifactKey("tokenizer"),
        model_id=manifest.embedding.model_id,
        dim=manifest.embedding.dim,
        session_options=session_options,
    )
    entropy = OnlineEntropy(seed=resolved.determinism.seed, as_of=online.as_of.date())
    # Never written to (this script stops at R7) -- bound for type-shape
    # parity with the real composition root, not exercised.
    submission_sink = CsvSubmissionSinkAdapter(
        Path(tempfile.gettempdir()) / "profile_submission_analytics.unused.csv",
        score_decimals=online.score_presentation.decimals,
    )
    run_report_sink = JsonRunReportSinkAdapter(
        Path(tempfile.gettempdir()) / "profile_submission_analytics.unused.json"
    )

    loaded = stages.r0_load(
        artifact_store=artifact_store,
        embedding_model=embedding_model,
        vector_store=vector_store,
        entropy=entropy,
        config=run_config,
    )
    return OnlineRunContext(
        config=run_config,
        candidate_source=candidate_source,
        vector_store=vector_store,
        embedding_model=embedding_model,
        submission_sink=submission_sink,
        run_report_sink=run_report_sink,
        entropy=entropy,
        artifact_store=artifact_store,
        artifacts=loaded.artifacts,
        manifest=loaded.manifest,
        as_of=entropy.as_of(),
        seed=entropy.seed,
        input_file_sha256=_sha256_file(input_path),
    )


def _run_real_stages(
    ctx: OnlineRunContext,
) -> tuple[stages.GatedSet, stages.ScoredSet, Any]:
    """R1 -> R2 -> R3 -> R4 -> R5 -> R6 -> R7, the real pipeline, nothing else."""
    ingested = stages.r1_ingest(ctx)
    featured = stages.r2_features(ctx, ingested)
    situated = stages.r3_semantic(ctx, featured)
    gated = stages.r4_gates(ctx, situated)
    scored = stages.r5_score(ctx, gated)
    ranking = stages.r6_rank(ctx, scored)
    ranking = stages.r7_reason(ctx, ranking, gated)
    return gated, scored, ranking


# --------------------------------------------------------------------------- #
# Step 3 -- per-candidate telemetry, straight off the real domain objects.   #
# --------------------------------------------------------------------------- #
def _matched_tokens(group: str, raw: Any) -> list[str]:
    """Literal lexicon tokens this candidate's skills/career text actually hit.

    Read-only diagnostic over the same ``_LEXICON_TOKENS``/``_tokenize`` the
    real ``features.skills.extract`` uses -- not a separate scoring path.
    """
    tokens = LEXICON_TOKENS[group]
    skill_tokens: set[str] = set()
    for skill in raw.skills:
        skill_tokens |= _tokenize(skill.name)
    career_tokens: set[str] = set()
    for position in raw.career_history:
        career_tokens |= _tokenize(position.description)
    return sorted(tokens & (skill_tokens | career_tokens))


def _competency_breakdown(cells: dict[str, Any], raw: Any) -> dict[str, Any]:
    """Per-group trust/in_career/semantic/competency + the real top-3 pick.

    ``stages._skill_match_value`` is called directly (not re-derived) so the
    reported skill_match always matches what the gate/scorer actually used.
    """
    groups: dict[str, dict[str, Any]] = {}
    for group in COMPETENCY_GROUPS:
        groups[group] = {
            "trust": float(cells[f"{group}.trust"].value),
            "in_career": float(cells[f"{group}.in_career"].value),
            "semantic": float(cells[f"{group}.semantic"].value),
            "competency": float(cells[f"{group}.competency"].value),
            "matched_tokens": _matched_tokens(group, raw),
        }
    ranked = sorted(groups.items(), key=lambda kv: kv[1]["competency"], reverse=True)
    top3 = [name for name, _ in ranked[:SKILL_MATCH_TOP_K]]
    skill_match = stages._skill_match_value(cells)
    # Restricted to the top-3 groups that actually drive skill_match -- every
    # candidate has *some* peripheral skill among the other six groups with
    # no career-text overlap (normal; nobody's job description mentions every
    # skill they've ever listed). Flagging that is noise. Flagging it on a
    # group that is one of the candidate's top-3 *score-driving* groups is
    # the real "claimed it, never evidenced it on the job, and it's carrying
    # real weight in the final score" signal.
    dabbler_groups = [
        name
        for name in top3
        if groups[name]["in_career"] <= 0.0
        and groups[name]["trust"] >= DABBLER_TRUST_FLOOR
    ]
    return {
        "groups": groups,
        "top3_groups": top3,
        "skill_match": skill_match,
        "dabbler_groups": dabbler_groups,
    }


def _eligibility_badges(report: EligibilityReport) -> list[str]:
    badges = [f"[\U0001f6ab HARD_BLOCK:{f.code.value}]" for f in report.hard_blocks]
    badges += [f"[⚠️ SOFT:{f.code.value}]" for f in report.soft_penalties]
    return badges


def build_candidate_record(
    rank_row: dict[str, str],
    gated: stages.GatedSet,
    scored_by_id: dict[CandidateId, ScoredCandidate],
    reasoning_by_id: dict[CandidateId, str],
    index_by_id: dict[CandidateId, int],
) -> dict[str, Any]:
    cid = CandidateId(rank_row["candidate_id"])
    idx = index_by_id[cid]
    ingested = gated.candidates[idx]
    raw = ingested.raw
    rep = gated.representations[idx]
    cells = dict(gated.cells[idx])
    scored = scored_by_id[cid]
    breakdown = scored.breakdown
    eligibility = rep.require_eligibility()
    career = rep.require_career()
    logistics = rep.require_logistics()
    behavioral = rep.require_behavioral()

    competency = _competency_breakdown(cells, raw)
    components = {
        cv.component.value: {
            "raw": float(cv.raw),
            "weight": cv.weight,
            "weighted": cv.weighted,
        }
        for cv in breakdown.components
    }

    badges = _eligibility_badges(eligibility)
    if not breakdown.integrity_gate.passed:
        badges.append("[\U0001f6ab HONEYPOT]")
    if float(breakdown.behavioral_multiplier) < 0.85:
        badges.append("[⚠️ GHOST_PENALTY]")
    if competency["dabbler_groups"]:
        dabbler_str = ",".join(competency["dabbler_groups"])
        badges.append(f"[\U0001f425 DABBLER_PATTERN:{dabbler_str}]")

    signals = raw.redrob_signals
    education = raw.education

    return {
        "rank": int(rank_row["rank"]),
        "score": float(rank_row["score"]),
        "candidate_id": cid,
        "name": raw.profile.anonymized_name,
        "current_title": raw.profile.current_title,
        "current_company": raw.profile.current_company,
        "location": f"{raw.profile.location}, {raw.profile.country}",
        "years_of_experience": raw.profile.years_of_experience,
        "derived_experience_years": float(career.derived_experience_years),
        "education_summary": (
            f"{education[0].degree} from {education[0].institution}"
            if education
            else "N/A"
        ),
        "components": components,
        "competency": competency,
        "eligibility": {
            "is_eligible": eligibility.is_eligible,
            "hard_blocks": [f.code.value for f in eligibility.hard_blocks],
            "soft_penalties": [f.code.value for f in eligibility.soft_penalties],
        },
        "integrity_gate_passed": breakdown.integrity_gate.passed,
        "behavioral_multiplier": float(breakdown.behavioral_multiplier),
        "logistics_multiplier": float(breakdown.logistics_multiplier),
        "logistics": {
            "location_fit": logistics.location_fit.value,
            "notice_fit": logistics.notice_fit.value,
            "notice_period_days": logistics.notice_period_days,
        },
        "behavioral": {
            "availability_status": behavioral.availability_status.value,
            "responsiveness_status": behavioral.responsiveness_status.value,
            "response_rate": signals.recruiter_response_rate,
            "github_activity_score": signals.github_activity_score,
            "expected_salary_min_max": [
                signals.expected_salary_range_inr_lpa.min,
                signals.expected_salary_range_inr_lpa.max,
            ],
        },
        "risk_badges": badges,
        "reasoning": reasoning_by_id.get(cid, ""),
    }


# --------------------------------------------------------------------------- #
# Step 4 -- the executive dashboard.                                          #
# --------------------------------------------------------------------------- #
def _load_pool_summary() -> dict[str, Any]:
    if not RUN_REPORT_PATH.is_file():
        return {}
    report: dict[str, Any] = json.loads(RUN_REPORT_PATH.read_text(encoding="utf-8"))
    summary: dict[str, Any] = report.get("reproducible", {})
    return summary


def render_dashboard(
    records: list[dict[str, Any]], pool_summary: dict[str, Any]
) -> str:
    lines: list[str] = []
    lines.append("# Top-100 Candidate Discovery -- Executive Triage Dashboard")
    lines.append("")
    lines.append(
        "Re-derived from the real `EligibilityEngine` / `ScoringEngine` / "
        "`features.skills` objects via `scripts/profile_submission_analytics.py` "
        "-- no parallel scoring math."
    )
    lines.append("")

    # --- Executive KPI section -------------------------------------------- #
    pool_count = pool_summary.get("candidate_count", "n/a")
    eligibility_summary: dict[str, int] = pool_summary.get("eligibility_summary", {})
    honeypot_top100 = sum(1 for r in records if not r["integrity_gate_passed"])
    hard_blocked_in_top100 = sum(
        1 for r in records if not r["eligibility"]["is_eligible"]
    )
    top10_yoe = [r["years_of_experience"] for r in records if r["rank"] <= 10]
    top10_avg_yoe = sum(top10_yoe) / len(top10_yoe) if top10_yoe else 0.0
    dabbler_count = sum(1 for r in records if r["competency"]["dabbler_groups"])
    ghost_count = sum(1 for r in records if r["behavioral_multiplier"] < 0.85)

    lines.append("## Executive KPI Summary")
    lines.append("")
    lines.append(f"- **Full pool size:** {pool_count}")
    lines.append(f"- **Top-10 cohort average YOE:** {top10_avg_yoe:.2f} years")
    lines.append(
        f"- **Active honeypots caught in top 100:** {honeypot_top100} / 100 "
        "(verified via `IntegrityEngine`)"
    )
    lines.append(
        f"- **Hard-blocked candidates leaking into top 100:** "
        f"{hard_blocked_in_top100} / 100 (should always be 0 -- a nonzero count "
        "means the eligible pool fell below 100 and the ranker padded with "
        "floored rows)"
    )
    lines.append(
        f"- **Self-learner / dabbler pattern flagged:** {dabbler_count} / 100 "
        f"(in_career == 0 with trust >= {DABBLER_TRUST_FLOOR} on a top-3 "
        "contributing group)"
    )
    lines.append(
        "- **Ghost-profile penalty (behavioral_multiplier < 0.85):** "
        f"{ghost_count} / 100"
    )
    lines.append("")

    if eligibility_summary:
        lines.append(
            "### Hard-Gate Fired Rates (full pool, from latest `run_report.json`)"
        )
        lines.append("")
        lines.append("| Gate | Fired | % of Pool |")
        lines.append("| :--- | ---: | ---: |")
        denom = pool_count if isinstance(pool_count, int) and pool_count else 1
        for code, count in sorted(eligibility_summary.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{code}` | {count:,} | {100 * count / denom:.2f}% |")
        lines.append("")

    # --- Triage table ------------------------------------------------------ #
    lines.append("## High-Density Triage Table")
    lines.append("")
    lines.append(
        "| Rank | Score | ID | Title @ Company | Top-3 Competency (skill_match) | "
        "Logistics & Availability | Gate / Risk Alerts |"
    )
    lines.append("| ---: | ---: | :--- | :--- | :--- | :--- | :--- |")
    for r in records:
        top3 = r["competency"]["top3_groups"]
        top3_parts = []
        for g in top3:
            gv = r["competency"]["groups"][g]
            tokens = ",".join(gv["matched_tokens"]) or "—"
            top3_parts.append(f"{g}={gv['competency']:.3f} [{tokens}]")
        comp_cell = "<br>".join(top3_parts) + (
            f"<br>**skill_match={r['competency']['skill_match']:.6f}**"
        )
        title_cell = (
            f"**{r['name']}**<br>`{r['current_title']}` @ "
            f"*{r['current_company']}*<br>YOE {r['years_of_experience']}"
        )
        log_cell = (
            f"{r['logistics']['location_fit']} / {r['logistics']['notice_fit']}<br>"
            f"notice {r['logistics']['notice_period_days']}d<br>"
            f"resp {r['behavioral']['response_rate']:.0%} "
            f"({r['behavioral']['responsiveness_status']})"
        )
        risk_cell = "<br>".join(r["risk_badges"]) if r["risk_badges"] else "[✅ CLEAR]"
        lines.append(
            f"| {r['rank']} | {r['score']:.6f} | `{r['candidate_id']}` | "
            f"{title_cell} | {comp_cell} | {log_cell} | {risk_cell} |"
        )
    lines.append("")

    # --- Reasoning manifest, separated -------------------------------------- #
    lines.append("## Model Generation Reasoning Manifest")
    lines.append("")
    lines.append("| Rank | ID | Score (6dp) | Reasoning |")
    lines.append("| ---: | :--- | ---: | :--- |")
    for r in records:
        lines.append(
            f"| {r['rank']} | `{r['candidate_id']}` | {r['score']:.6f} | "
            f"*\"{r['reasoning']}\"* |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main.                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    print("[*] Reading submission.csv ...")
    rank_rows = _read_submission_rows(SUBMISSION_PATH)
    target_ids = {row["candidate_id"] for row in rank_rows}

    print(f"[*] Streaming {DATASET_PATH.name} for {len(target_ids)} candidates ...")
    target_lines = _extract_target_lines(DATASET_PATH, target_ids)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as handle:
        for cid in (row["candidate_id"] for row in rank_rows):
            handle.write(target_lines[cid])
        temp_input = Path(handle.name)

    try:
        print("[*] Re-running the real R0-R7 online stages on the submitted slice ...")
        ctx = _build_context(temp_input)
        gated, scored, ranking = _run_real_stages(ctx)
    finally:
        temp_input.unlink(missing_ok=True)

    index_by_id = {c.candidate_id: i for i, c in enumerate(gated.candidates)}
    scored_by_id = {s.candidate_id: s for s in scored.scored}
    reasoning_by_id = {
        ranked.candidate_id: (
            "" if ranked.reasoning is None else ranked.reasoning.rendered
        )
        for ranked in ranking.ordered
    }

    print("[*] Building per-candidate telemetry ...")
    records = [
        build_candidate_record(row, gated, scored_by_id, reasoning_by_id, index_by_id)
        for row in rank_rows
    ]
    records.sort(key=lambda r: r["rank"])

    print(f"[*] Writing {LEAN_JSON_PATH} ...")
    LEAN_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEAN_JSON_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[*] Writing {DASHBOARD_PATH} ...")
    pool_summary = _load_pool_summary()
    DASHBOARD_PATH.write_text(render_dashboard(records, pool_summary), encoding="utf-8")

    print("[==========] Analytics suite complete.")


if __name__ == "__main__":
    main()
