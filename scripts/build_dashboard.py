
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEAN_PATH = REPO / "artifacts" / "debug_top100_lean.json"
RAW_CANDIDATES_PATH = REPO / "data" / "raw" / "candidates.jsonl"
RUN_REPORT_PATH = REPO / "artifacts" / "run_report.json"
TEMPLATE_PATH = Path(__file__).resolve().parent / "dashboard_template.html"
OUT_PATH = REPO / "artifacts" / "debug_dashboard.html"

GROUP_LABELS = {
    "llm": "LLM / GenAI", "ir": "Information Retrieval", "mlops": "MLOps",
    "mle": "ML Engineering", "nlp": "NLP", "rank": "Ranking", "retr": "Retrieval Systems",
    "recsys": "Recommender Systems", "eval": "Eval / Metrics",
}
COMPONENT_LABELS = {
    "skill_match": "Skill Match", "semantic_fit": "Semantic Fit", "career_fit": "Career Trajectory Fit",
    "experience_fit": "Experience Band Fit", "education_fit": "Education Fit",
    "credibility": "Credibility", "archetype_fit": "Archetype Fit",
}
GATE_LABELS = {
    "primary_cv_speech_robotics_no_nlp": "Primary CV speech/robotics, no NLP",
    "notice_over_30": "Notice period over 30 days",
    "pure_research_no_production": "Pure research, no production experience",
    "outside_india_no_sponsor": "Outside India, no sponsorship path",
    "title_chaser_sub_18m_hops": "Title-chaser: sub-18-month hops",
    "consulting_firms_only_career": "Consulting firms only career",
    "closed_source_5y_no_validation": "Closed-source 5y, no validation",
    "outside_experience_band": "Outside target experience band",
}
RISK_DESCRIPTIONS = {
    "title_chaser_sub_18m_hops": "Frequent sub-18-month role changes detected across the career history — a job-hopping signal that may indicate limited depth per role.",
    "notice_over_30": "Notice period exceeds the 30-day ideal availability window.",
    "outside_india_no_sponsor": "Candidate is based outside India without an active sponsorship path on file.",
    "outside_experience_band": "Years of experience fall outside the target band for this role.",
    "pure_research_no_production": "Career history skews toward pure research with no shipped production work.",
    "consulting_firms_only_career": "Entire career history is at consulting firms only.",
    "closed_source_5y_no_validation": "5+ years on closed-source work with no externally validated artifacts.",
    "primary_cv_speech_robotics_no_nlp": "Primary CV is speech/robotics-flavored with no NLP corroboration.",
}


def build_candidate(d: dict) -> dict:
    raw = d.get("raw", {})
    profile = raw.get("profile", {})
    elig = d["eligibility"]
    soft = set(elig.get("soft_penalties", []))
    dabbler_groups = d["competency"].get("dabbler_groups", [])
    ghost = d["behavioral_multiplier"] < 0.85
    flags = {
        "clear": not soft and not dabbler_groups and not ghost and elig["is_eligible"],
        "title_chaser": "title_chaser_sub_18m_hops" in soft,
        "dabbler": bool(dabbler_groups),
        "ghost": ghost,
        "notice_over_30": "notice_over_30" in soft,
        "outside_india": "outside_india_no_sponsor" in soft,
    }

    search_terms = [
        d["name"], d["candidate_id"], d["current_title"], d["current_company"],
        d.get("location", ""), profile.get("headline", ""),
    ]
    for g in d["competency"]["groups"].values():
        search_terms.extend(g.get("matched_tokens", []))
    for ch in raw.get("career_history", []):
        search_terms.extend([ch.get("company", ""), ch.get("title", "")])
    search_blob = " ".join(str(t) for t in search_terms).lower()

    risk_items = []
    for gate in elig.get("hard_blocks", []):
        risk_items.append({"level": "hard", "code": gate, "desc": RISK_DESCRIPTIONS.get(gate, gate)})
    for gate in elig.get("soft_penalties", []):
        risk_items.append({"level": "soft", "code": gate, "desc": RISK_DESCRIPTIONS.get(gate, gate)})
    for grp in dabbler_groups:
        risk_items.append({
            "level": "dabbler", "code": f"dabbler:{grp}",
            "desc": f"Self-learner / dabbler pattern in {GROUP_LABELS.get(grp, grp)} -- trust score >= 0.5 with zero in-career corroboration (claims skill, never used it on the job).",
        })
    if ghost:
        risk_items.append({
            "level": "ghost", "code": "ghost_penalty",
            "desc": f"Ghost-profile penalty applied -- behavioral multiplier {d['behavioral_multiplier']:.3f} is below the 0.85 trust threshold (low availability/responsiveness signal).",
        })
    if not raw.get("redrob_signals", {}).get("verified_email", True):
        risk_items.append({"level": "soft", "code": "unverified_email", "desc": "Email is not verified on the platform."})

    return {
        "rank": d["rank"], "score": d["score"], "candidate_id": d["candidate_id"], "name": d["name"],
        "current_title": d["current_title"], "current_company": d["current_company"],
        "location": d.get("location", ""), "yoe": d["years_of_experience"],
        "derived_yoe": d["derived_experience_years"], "education_summary": d.get("education_summary", ""),
        "components": d["components"], "competency": d["competency"],
        "eligibility": elig, "integrity_gate_passed": d.get("integrity_gate_passed", True),
        "behavioral_multiplier": d["behavioral_multiplier"], "logistics_multiplier": d["logistics_multiplier"],
        "logistics": d["logistics"], "behavioral": d["behavioral"], "risk_badges": d.get("risk_badges", []),
        "reasoning": d["reasoning"], "flags": flags, "search_blob": search_blob, "risk_items": risk_items,
        "profile": {
            "headline": profile.get("headline", ""), "summary": profile.get("summary", ""),
            "location": profile.get("location", ""), "country": profile.get("country", ""),
            "company_size": profile.get("current_company_size", ""), "industry": profile.get("current_industry", ""),
        },
        "career_history": raw.get("career_history", []),
        "education_detail": raw.get("education", []),
        "certifications": raw.get("certifications", []),
        "languages": raw.get("languages", []),
        "redrob": raw.get("redrob_signals", {}),
    }


def load_raw_records(candidate_ids: set) -> dict:

    found = {}
    with RAW_CANDIDATES_PATH.open() as f:
        for line in f:
            if '"candidate_id"' not in line:
                continue
            record = json.loads(line)
            cid = record.get("candidate_id")
            if cid in candidate_ids:
                found[cid] = record
                if len(found) == len(candidate_ids):
                    break
    missing = candidate_ids - found.keys()
    if missing:
        raise RuntimeError(f"candidates.jsonl is missing {len(missing)} expected ids: {sorted(missing)[:5]}...")
    return found


def main():
    lean = json.loads(LEAN_PATH.read_text())
    run_report = json.loads(RUN_REPORT_PATH.read_text())

    candidate_ids = {d["candidate_id"] for d in lean}
    raw_by_id = load_raw_records(candidate_ids)
    merged = [{**d, "raw": raw_by_id[d["candidate_id"]]} for d in lean]

    candidates = [build_candidate(d) for d in merged]
    candidates.sort(key=lambda c: c["rank"])

    pool_size = run_report["reproducible"]["candidate_count"]
    top10 = [c for c in candidates if c["rank"] <= 10]
    avg_yoe_top10 = sum(c["yoe"] for c in top10) / len(top10)
    honeypot_count = sum(1 for c in candidates if not c["integrity_gate_passed"])
    hard_blocked_count = sum(1 for c in candidates if not c["eligibility"]["is_eligible"])
    dabbler_count = sum(1 for c in candidates if c["flags"]["dabbler"])
    ghost_count = sum(1 for c in candidates if c["flags"]["ghost"])

    gate_summary = run_report["reproducible"]["eligibility_summary"]
    gates = sorted(
        ({"code": k, "label": GATE_LABELS.get(k, k), "fired": v, "pct": v / pool_size * 100}
         for k, v in gate_summary.items()),
        key=lambda g: -g["fired"],
    )

    kpis = {
        "pool_size": pool_size, "avg_yoe_top10": round(avg_yoe_top10, 2),
        "honeypot_count": honeypot_count, "hard_blocked_count": hard_blocked_count,
        "dabbler_count": dabbler_count, "ghost_count": ghost_count, "cohort_size": len(candidates),
    }

    run_meta = {
        "run_id": run_report["audit"]["run_id"], "code_version": run_report["reproducible"]["code_version"],
        "manifest_hash": run_report["reproducible"]["manifest_hash"][:16],
        "reproducible": run_report["budget"]["within_budget"],
    }

    payload = {
        "kpis": kpis, "gates": gates, "candidates": candidates,
        "group_labels": GROUP_LABELS, "component_labels": COMPONENT_LABELS, "run_meta": run_meta,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</script>", "<\\/script>")

    template = TEMPLATE_PATH.read_text()
    html = template.replace("__DATA_JSON__", data_json)
    OUT_PATH.write_text(html)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes), {len(candidates)} candidates")


if __name__ == "__main__":
    main()
