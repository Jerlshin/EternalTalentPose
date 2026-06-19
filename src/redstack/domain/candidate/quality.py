

from __future__ import annotations

from typing import final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from redstack.domain.errors import CQVInvariantError
from redstack.domain.ids import FeatureIndex


@final
class FeatureLayoutEntry(BaseModel):
    """One ordered feature slot: name, index, source slice, and bounds."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
    )

    name: str = Field(min_length=1)
    index: FeatureIndex = Field(ge=0)
    source_slice: str = Field(min_length=1)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)


@final
class FeatureLayout(BaseModel):
    """Ordered, versioned CQV index map — the single source of feature order."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    entries: tuple[FeatureLayoutEntry, ...] = Field(min_length=1)
    layout_version: str = Field(min_length=1)

    @property
    def dim(self) -> int:
        return len(self.entries)

    @field_validator("entries", mode="after")
    @classmethod
    def _ordered_contiguous_unique(
        cls, value: tuple[FeatureLayoutEntry, ...]
    ) -> tuple[FeatureLayoutEntry, ...]:
        names: set[str] = set()
        for position, entry in enumerate(value):
            if int(entry.index) != position:
                raise ValueError(
                    "FeatureLayout indices must be contiguous 0..D-1 in order"
                )
            if entry.lower > entry.upper:
                raise ValueError("FeatureLayout entry lower bound exceeds upper bound")
            if entry.name in names:
                raise ValueError("FeatureLayout feature names must be unique")
            names.add(entry.name)
        return value


@final
class CandidateQualityVector(BaseModel):
    """The fixed-length float32 feature vector dotted with ``ScoringWeights``."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True, arbitrary_types_allowed=True
    )

    values: npt.NDArray[np.float32]
    schema_version: str = Field(min_length=1)

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_finite_immutable(cls, value: object) -> npt.NDArray[np.float32]:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 1:
            raise CQVInvariantError("CQV values must be a 1-D array")
        if not np.all(np.isfinite(array)):
            raise CQVInvariantError("CQV values must contain no NaN/inf")
        frozen = np.ascontiguousarray(array, dtype=np.float32).copy()
        frozen.setflags(write=False)
        return frozen

    @classmethod
    def create(
        cls,
        *,
        values: npt.NDArray[np.float32],
        schema_version: str,
        layout: FeatureLayout,
    ) -> CandidateQualityVector:
        """Build a CQV validated against ``layout`` (dim, per-index bounds).

        The layout is consumed transiently and not retained on the instance.
        """
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != layout.dim:
            raise CQVInvariantError(
                f"CQV dim mismatch: got {array.shape}, expected ({layout.dim},)"
            )
        if not np.all(np.isfinite(array)):
            raise CQVInvariantError("CQV values must contain no NaN/inf")
        for entry in layout.entries:
            cell = float(array[int(entry.index)])
            if not (entry.lower <= cell <= entry.upper):
                raise CQVInvariantError(
                    f"CQV cell {entry.name} out of bounds "
                    f"[{entry.lower}, {entry.upper}]"
                )
        return cls(values=array, schema_version=schema_version)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidateQualityVector):
            return NotImplemented
        return self.schema_version == other.schema_version and np.array_equal(
            self.values, other.values
        )

    __hash__ = None  # type: ignore[assignment]


__all__: tuple[str, ...] = (
    "CandidateQualityVector",
    "FeatureLayout",
    "FeatureLayoutEntry",
)