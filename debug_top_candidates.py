#!/usr/bin/env python3
"""High-Density Candidate Triage & Debugging Utility.

Reads submission.csv, extracts matching raw candidates from candidates.jsonl,
filters out heavy text paragraphs, computes defensive triage signals (keyword
stuffing, response rates, location alignment), and exports two clean formats:
1. artifacts/debug_top100_lean.json (Structured metrics only)
2. artifacts/debug_dashboard.md      (Interactive triage dashboard)
"""

import csv
import json
from pathlib import Path
import sys

# Define core AI skills requested by the JD to track keyword stuffing
CORE_AI_SKILLS = {
    "embeddings", "retrieval", "ranking", "llms", "fine-tuning", "rag",
    "learning to rank", "peft", "lora", "qlora", "pinecone", "weaviate",
    "milvus", "qdrant", "vector search", "hybrid retrieval", "xgboost"
}

# Configuration Paths
SUBMISSION_PATH = Path("artifacts/submission.csv")
DATASET_PATH = Path("data/raw/candidates.jsonl")
LEAN_JSON_PATH = Path("artifacts/debug_top100_lean.json")
DASHBOARD_PATH = Path("artifacts/debug_dashboard.md")


def parse_ranked_ids(path: Path) -> list[dict]:
    """Load candidate ids, ranks, and scores preserving exact ordering."""
    records = []
    if not path.is_file():
        print(f"[-] Error: Submission file missing at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("candidate_id"):
                records.append({
                    "candidate_id": row["candidate_id"].strip(),
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "reasoning": row.get("reasoning", "").strip()
                })
    return records


def stream_raw_profiles(path: Path, target_ids: set) -> dict:
    """Stream pool to gather raw entries for target IDs with zero copy."""
    profiles = {}
    if not path.is_file():
        print(f"[-] Error: Dataset missing at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # Quick substring pre-filter to bypass heavy JSON parsing overhead
            if any(tid in line for tid in target_ids):
                record = json.loads(line)
                cid = record.get("candidate_id")
                if cid in target_ids:
                    profiles[cid] = record
                    if len(profiles) == len(target_ids):
                        break
    return profiles


def compute_telemetry(rank_meta: dict, raw: dict) -> dict:
    """Filter raw attributes and calculate defensive triage flags."""
    profile = raw.get("profile", {})
    signals = raw.get("redrob_signals", {})
    history = raw.get("career_history", [])
    education = raw.get("education", [])
    skills = [s.get("name", "").lower() for s in raw.get("skills", [])]

    # Compute skill-stuffing parameters
    ai_skills_found = [s for s in skills if s in CORE_AI_SKILLS]
    stuffing_ratio = len(ai_skills_found) / len(skills) if skills else 0.0

    # Extract clean company history profile
    companies = [c.get("company", "Unknown") for c in history]
    titles = [c.get("title", "Unknown") for c in history]
    
    # Location Verification
    loc_string = f"{profile.get('location', '')}, {profile.get('country', '')}"
    is_hub = any(h in loc_string.lower() for h in ["pune", "noida", "hyderabad", "mumbai", "delhi"])
    willing_relocate = signals.get("willing_to_relocate", False)

    # Core Lean Representation Object
    return {
        "rank": rank_meta["rank"],
        "score": rank_meta["score"],
        "candidate_id": raw["candidate_id"],
        "name": profile.get("anonymized_name"),
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "years_of_experience": profile.get("years_of_experience"),
        "location": loc_string,
        "education_summary": f"{education[0].get('degree', '')} from {education[0].get('institution', '')}" if education else "N/A",
        "behavioral": {
            "response_rate": signals.get("recruiter_response_rate"),
            "last_active": signals.get("last_active_date"),
            "notice_period_days": signals.get("notice_period_days"),
            "github_score": signals.get("github_activity_score"),
            "expected_salary_min_max": [
                signals.get("expected_salary_range_inr_lpa", {}).get("min"),
                signals.get("expected_salary_range_inr_lpa", {}).get("max")
            ]
        },
        "telemetry": {
            "total_skills_count": len(skills),
            "ai_skills_count": len(ai_skills_found),
            "ai_skills_ratio": round(stuffing_ratio, 2),
            "career_trajectory_titles": titles[:3],
            "career_trajectory_companies": companies[:3]
        },
        "risk_flags": {
            "low_availability_risk": signals.get("recruiter_response_rate", 1.0) < 0.40,
            "keyword_stuffer_risk": stuffing_ratio > 0.65 and "engineer" not in profile.get("current_title", "").lower(),
            "location_mismatch": not (is_hub or willing_relocate),
            "consulting_heavy": all(any(k in c.lower() for k in ["services", "consulting", "tcs", "infosys", "wipro", "cognizant"]) for c in companies if c) if companies else False
        },
        "submission_reasoning": rank_meta["reasoning"]
    }


def generate_markdown_dashboard(candidates: list[dict], path: Path) -> None:
    """Compile an interactive high-density Markdown debugging dashboard."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 💎 Top 100 Ranked Candidate Discovery Dashboard\n")
        f.write(" Use this dashboard to instantly audit profile fit, track signal anomalies, and audit for honeypots.\n\n")
        
        # Risk Flags KPI Section
        stuffer_cnt = sum(1 for c in candidates if c["risk_flags"]["keyword_stuffer_risk"])
        unavail_cnt = sum(1 for c in candidates if c["risk_flags"]["low_availability_risk"])
        loc_cnt = sum(1 for c in candidates if c["risk_flags"]["location_mismatch"])
        
        f.write("### 🚨 Top 100 Risk Monitoring Guardrails\n")
        f.write(f"* **Detected Keyword Stuffers:** {stuffer_cnt} / 100\n")
        f.write(f"* **Low Availability / Ghost Risks:** {unavail_cnt} / 100\n")
        f.write(f"* **Out of Hub / Relocation Risks:** {loc_cnt} / 100\n\n")
        
        # Comprehensive Triage Table
        f.write("### 📊 Candidate Triage Roster\n")
        f.write("| Rank | Score | Candidate ID | Profile Details | Experience & Domain Fit | Availability & Signals | Risk Alerts |\n")
        f.write("| :---: | :---: | :---: | :--- | :--- | :--- | :--- |\n")
        
        for c in candidates:
            # Format columns beautifully
            profile_str = f"**{c['name']}**<br>`{c['current_title']}` at *{c['current_company']}*<br>📍 {c['location']}"
            exp_str = f"**YOE:** {c['years_of_experience']} yrs<br>🎓 {c['education_summary']}<br>🧩 **AI Skills:** {c['telemetry']['ai_skills_count']}/{c['telemetry']['total_skills_count']} ({int(c['telemetry']['ai_skills_ratio']*100)}%)"
            
            b = c["behavioral"]
            sig_str = f"📥 **Response Rate:** {int(b['response_rate']*100)}%<br>⏰ **Notice:** {b['notice_period_days']} days<br>💻 **GitHub Score:** {b['github_score'] if b['github_score'] != -1 else 'N/A'}<br>💰 **Expected:** {b['expected_salary_min_max'][0]}-{b['expected_salary_min_max'][1]} LPA"
            
            # Aggregate Risk Status Marks
            flags = []
            if c["risk_flags"]["keyword_stuffer_risk"]: flags.append("⚠️ **KEYWORD_STUFFER**")
            if c["risk_flags"]["low_availability_risk"]: flags.append("🛑 **GHOST_PROFILE**")
            if c["risk_flags"]["location_mismatch"]: flags.append("📍 **LOCATION_MISMATCH**")
            if c["risk_flags"]["consulting_heavy"]: flags.append("🏢 *CONSULTING_ONLY*")
            risk_str = "<br>".join(flags) if flags else "✅ Clear"
            
            f.write(f"| {c['rank']} | {c['score']:.4f} | `{c['candidate_id']}` | {profile_str} | {exp_str} | {sig_str} | {risk_str} |\n")
            
        f.write("\n\n### 📝 Model Generation Reasoning Manifest\n")
        f.write("| Rank | Candidate ID | Model Reasoning Output |\n")
        f.write("| :---: | :---: | :--- |\n")
        for c in candidates:
            f.write(f"| {c['rank']} | `{c['candidate_id']}` | *\"{c['submission_reasoning']}\"* |\n")


def main():
    print("[*] Launching high-density telemetry extractor...")
    submission_meta = parse_ranked_ids(SUBMISSION_PATH)
    target_ids = {m["candidate_id"] for m in submission_meta}
    
    print(f"[*] Extracting raw profile snapshots for {len(target_ids)} candidates...")
    raw_profiles = stream_raw_profiles(DATASET_PATH, target_ids)
    
    # Process and sort back into precise ranking order
    curated_candidates = []
    for meta in submission_meta:
        cid = meta["candidate_id"]
        if cid in raw_profiles:
            telemetry_node = compute_telemetry(meta, raw_profiles[cid])
            curated_candidates.append(telemetry_node)
            
    # Export structural lean JSON (Compressed from 20,000 lines down to ~3,000 lines)
    print(f"[*] Exporting filtered structural dataset to: {LEAN_JSON_PATH}")
    LEAN_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEAN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(curated_candidates, f, indent=2, ensure_ascii=False)
        
    # Export interactive Markdown Dashboard
    print(f"[*] Exporting visual triage dashboard to: {DASHBOARD_PATH}")
    generate_markdown_dashboard(curated_candidates, DASHBOARD_PATH)
    
    print("[==========] Extraction Complete. Debug formats successfully serialized.")


if __name__ == "__main__":
    main()