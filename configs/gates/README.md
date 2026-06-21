# `configs/gates/` — Eligibility Rules

[`eligibility_rules.yaml`](eligibility_rules.yaml) is the human-authored, declarative expression of the job description's explicit disqualifiers and penalties — every code here is cross-checked at build time against `redstack.domain.enums.EligibilityCode`, and the severity classification (hard vs. soft) mirrors `src/redstack/engines/eligibility.py`'s own classification exactly.

## Hard blocks

A hard block makes a candidate **ineligible** — `EligibilityEngine` floors their score regardless of how well they otherwise match. No penalty weight applies; the candidate simply cannot rank.

| Code | Description |
|---|---|
| `pure_research_no_production` | Research-aligned profile with no production-company tenure |
| `langchain_openai_only_recent` | Recent framework-name claims without corroborated depth |
| `no_production_code_18m` | No production-company role within the recency window |
| `consulting_firms_only_career` | Entire career at consulting firms; no product-side delivery |
| `primary_cv_speech_robotics_no_nlp` | Primary expertise in an adjacent vision/speech/robotics domain |
| `closed_source_5y_no_validation` | Long closed-source tenure with no externally validated competency |

## Soft penalties

A soft penalty **down-weights** a candidate's score by the listed weight; it never floors them.

| Code | Description | Penalty |
|---|---|---|
| `title_chaser_sub_18m_hops` | High hop rate: many sub-18-month tenures | 0.15 |
| `notice_over_30` | Notice period exceeds 30 days | 0.10 |
| `outside_india_no_sponsor` | Located outside India without a sponsorship path | 0.15 |
| `outside_experience_band` | Derived experience outside the job description's target band | 0.10 |

## Pipeline

Offline stage **O6** packages this file into `artifacts/gates/eligibility_rules.yaml`, the frozen artifact `EligibilityEngine` actually evaluates online (stage R4). Editing this file alone has no effect on a ranking run until `redstack build` is re-run.

See [`/ARCHITECTURE.md` §6](../../ARCHITECTURE.md#6-the-engines) (`EligibilityEngine`).
