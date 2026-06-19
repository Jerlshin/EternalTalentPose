
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, final

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from redstack.domain.enums import EvidenceKind
from redstack.domain.ids import CandidateId

if TYPE_CHECKING:
    from redstack.domain.source import RawCandidate


@final
class EvidenceRef(BaseModel):
    """A typed pointer into a real value inside a ``RawCandidate``."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
    )

    kind: EvidenceKind
    path: str
    value: StrictStr | StrictBool | StrictInt | StrictFloat
    as_of: date | None = None


@final
class ProvenanceHandle(BaseModel):
    """How a representation refers back to its source without 100K x memory cost.

    Exactly one of ``inline`` (survivors / top-K) or ``source_index`` (bulk
    columnar path) is non-null.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True, defer_build=True
    )

    candidate_id: CandidateId
    inline: RawCandidate | None
    source_index: int | None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ProvenanceHandle:
        has_inline = self.inline is not None
        has_index = self.source_index is not None
        if has_inline == has_index:
            raise ValueError(
                "ProvenanceHandle requires exactly one of inline / source_index"
            )
        if self.inline is not None and self.inline.candidate_id != self.candidate_id:
            raise ValueError("ProvenanceHandle.inline.candidate_id mismatch")
        return self


__all__: tuple[str, ...] = ("EvidenceRef", "ProvenanceHandle")