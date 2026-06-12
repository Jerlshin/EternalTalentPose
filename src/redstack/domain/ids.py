"""REDSTACK Domain Layer — Nominal type separations checked statically via mypy."""
from typing import NewType

__all__ = [
    "CandidateId", "AnchorId", "ArchetypeId", "SkillName",
    "Score", "Similarity", "UnitScore", "Multiplier", "Months", "LpaAmount"
]

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
