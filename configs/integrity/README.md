# `configs/integrity/` — Honeypot Rule Shapes

[`honeypot_rules.yaml`](honeypot_rules.yaml) is the human-declared **shape** of each integrity ("honeypot") detection rule — which kind of impossibility it detects and whether it is, by nature, a hard or soft signal. It does not contain the calibrated numeric thresholds; those are data-driven, set by the offline build against the observed candidate pool.

## Current rules

| Flag | Severity | Detects |
|---|---|---|
| `EXPERIENCE_EXCEEDS_COMPANY_AGE` | hard | Claimed tenure at an employer exceeds that employer's age |
| `SKILL_DURATION_EXCEEDS_EXPERIENCE` | hard | A single skill's claimed duration exceeds total career experience |
| `EXPERT_AT_ZERO_MONTHS` | hard | Expert proficiency claimed with zero months and zero endorsements |
| `OVERLAPPING_FULL_TIME_POSITIONS` | soft | Concurrent full-time positions with implausible time overlap |
| `KEYWORD_STUFFING` | soft | Dense skill array with no corroborating evidence in descriptions |

## Pipeline

Offline stage **O3 (Honeypot Discovery)** calibrates the actual cut-point thresholds for these rules against the dataset census (stage O0), so a threshold is set relative to the observed distribution rather than hard-coded. The result is written to `artifacts/calibration/integrity_thresholds.json`, which `IntegrityEngine` evaluates online (stage R4).

A candidate is only floored as a honeypot when **two or more hard rules fire, or the calibrated composite risk score crosses threshold** — a single soft anomaly (e.g. one overlapping role) only dampens the score, by design, to avoid floor­ing real candidates on noisy-but-possible data.

See [`/ARCHITECTURE.md` §1](../../ARCHITECTURE.md#1-what-redstack-does) (integrity gating) and [`/ARCHITECTURE.md` §6](../../ARCHITECTURE.md#6-the-engines) (`IntegrityEngine`).
