"""``SemanticProfile`` slice — dense fit vs JD anchors (R3).

Owner layer: domain.
Allowed imports: ids, pydantic.

References (never inlines) the candidate vector via ``VectorRef``; the 384-d
array lives in the mmap'd store behind a port.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from redstack.domain.ids import AnchorId, Similarity, UnitScore

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


@final
class VectorRef(BaseModel):
    """An index into the mmap'd candidate/anchor vector store."""

    model_config = _STRICT

    store_key: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    dim: int = Field(gt=0)


@final
class SemanticProfile(BaseModel):
    """Anchor cosine signals + a reference to the dense vector."""

    model_config = _STRICT

    anchor_similarities: Mapping[AnchorId, Similarity]
    positive_fit: Similarity = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    negative_fit: Similarity = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    net_semantic_fit: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    best_positive_anchor: AnchorId | None
    vector_ref: VectorRef

    @field_validator("anchor_similarities", mode="after")
    @classmethod
    def _freeze_similarities(
        cls, value: Mapping[AnchorId, Similarity]
    ) -> Mapping[AnchorId, Similarity]:
        for sim in value.values():
            if not (-1.0 <= sim <= 1.0):
                raise ValueError("anchor similarity out of range [-1, 1]")
        return MappingProxyType(dict(value))

    @field_serializer("anchor_similarities")
    def _dump_similarities(
        self, value: Mapping[AnchorId, Similarity]
    ) -> dict[AnchorId, Similarity]:
        return dict(value)

    @model_validator(mode="after")
    def _best_anchor_present(self) -> SemanticProfile:
        if (
            self.best_positive_anchor is not None
            and self.best_positive_anchor not in self.anchor_similarities
        ):
            raise ValueError("best_positive_anchor must be a known anchor")
        return self


__all__: tuple[str, ...] = ("SemanticProfile", "VectorRef")