
from __future__ import annotations

from typing import NewType

CandidateId = NewType("CandidateId", str)
AnchorId = NewType("AnchorId", str)
ArchetypeId = NewType("ArchetypeId", int)
SkillName = NewType("SkillName", str)
Score = NewType("Score", float)
Similarity = NewType("Similarity", float)
UnitScore = NewType("UnitScore", float)
Multiplier = NewType("Multiplier", float)
Months = NewType("Months", int)
LpaAmount = NewType("LpaAmount", float)
FeatureIndex = NewType("FeatureIndex", int)

#: Canonical candidate-id pattern (validator ``validate_submission.py`` law).
CANDIDATE_ID_PATTERN: str = r"^CAND_[0-9]{7}$"

__all__: tuple[str, ...] = (
    "CANDIDATE_ID_PATTERN",
    "AnchorId",
    "ArchetypeId",
    "CandidateId",
    "FeatureIndex",
    "LpaAmount",
    "Months",
    "Multiplier",
    "Score",
    "Similarity",
    "SkillName",
    "UnitScore",
)