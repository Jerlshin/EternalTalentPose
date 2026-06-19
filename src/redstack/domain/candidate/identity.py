

from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.ids import CANDIDATE_ID_PATTERN, CandidateId
from redstack.domain.provenance import ProvenanceHandle


@final
class Identity(BaseModel):
    """Candidate identity; the sole identity source for scored/ranked outputs."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
    )

    candidate_id: CandidateId = Field(pattern=CANDIDATE_ID_PATTERN)
    anonymized_name: str = Field(min_length=1)
    provenance: ProvenanceHandle

    @model_validator(mode="after")
    def _provenance_matches(self) -> Identity:
        if self.provenance.candidate_id != self.candidate_id:
            raise ValueError("provenance.candidate_id must equal candidate_id")
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Identity):
            return NotImplemented
        return self.candidate_id == other.candidate_id

    def __hash__(self) -> int:
        return hash(self.candidate_id)


__all__: tuple[str, ...] = ("Identity",)