"""``CQVAssembler`` — fold feature cells into the CQV (``engines/cqv.py``).

Owner layer: engines. Allowed imports: ``domain``, ``features``, ``config.schema``,
numpy. Forbidden: ``adapters``, ``pipelines``, ``observability``, sibling engines,
clock, online RNG.

Logical→physical (Repo §9): the ``CQVAssembler`` folds every populated slice's
feature values — surfaced uniformly through the read-only ``FeatureView`` — into
the fixed-length ``(D,)`` ``CandidateQualityVector`` aligned to ``FEATURE_LAYOUT``.
It is the boundary between rich typed profiles and vectorized scoring.

Determinism & invariants (Domain §G.10 / Online R5 / Testing §131):
- reduction strictly in ``FEATURE_LAYOUT`` index order; dtype ``float32``;
- ``len(values) == D``; **no ``NaN``/``inf``** (all sentinels resolved upstream);
- each index within its documented ``bounds``; ``schema_version`` == layout version.
Any violation raises ``CQVInvariantError`` (a programming/contract error, never a
business verdict).

Two surfaces: ``assemble`` materializes one ``CandidateQualityVector`` (survivors /
top-K); ``fold_row`` writes directly into a shared pre-allocated ``(N, D)`` matrix
at a deterministic row index (the 100K bulk path, no per-candidate object churn).
"""

from __future__ import annotations

import math
from typing import Final, final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from redstack.domain.candidate.quality import CandidateQualityVector
from redstack.domain.errors import CQVInvariantError
from redstack.features.layout import FEATURE_LAYOUT, layout_version
from redstack.features.view import FeatureView

# Number of features in the frozen layout (CQV dimensionality D).
_DIM: Final[int] = len(FEATURE_LAYOUT)
# Float comparison slack for bound checks (float32 round-trip tolerance).
_BOUND_EPS: Final[float] = 1e-6


@final
class CQVAssembler(BaseModel):
    """Stateless, pure folder from ``FeatureView`` to ``CandidateQualityVector``."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    @property
    def dim(self) -> int:
        """The frozen CQV dimensionality ``D``."""
        return _DIM

    # ------------------------------------------------------------------ public
    def assemble(self, view: FeatureView) -> CandidateQualityVector:
        """Fold one candidate's cells into a validated ``CandidateQualityVector``."""
        values: npt.NDArray[np.float32] = np.empty(_DIM, dtype=np.float32)
        self._fill(values, view)
        values.setflags(write=False)  # §N: inlined arrays are non-aliasable
        return CandidateQualityVector(values=values, schema_version=layout_version)

    def fold_row(
        self, matrix: npt.NDArray[np.float32], row: int, view: FeatureView
    ) -> None:
        """Write one candidate's folded values into row ``row`` of the bulk matrix.

        The matrix is the pipeline-owned ``(N, D)`` float32 array; this method does
        not allocate. Validation (finiteness, bounds) is identical to ``assemble``.
        """
        if matrix.dtype != np.float32:
            raise CQVInvariantError(
                f"bulk CQV matrix dtype {matrix.dtype!r}, expected float32"
            )
        if matrix.ndim != 2 or matrix.shape[1] != _DIM:
            raise CQVInvariantError(
                f"bulk CQV matrix shape {matrix.shape!r}, expected (N, {_DIM})"
            )
        if not (0 <= row < matrix.shape[0]):
            raise CQVInvariantError(f"row {row} out of range for {matrix.shape[0]} rows")
        self._fill(matrix[row], view)

    # --------------------------------------------------------------- internals
    @staticmethod
    def _fill(target: npt.NDArray[np.float32], view: FeatureView) -> None:
        """Populate ``target`` (length ``D``) from the view in layout order."""
        for entry in FEATURE_LAYOUT:
            feature_id = entry.feature_id
            if not view.has(feature_id):
                raise CQVInvariantError(
                    f"feature {feature_id!r} absent from view at fold time"
                )
            value = view.get(feature_id).value
            if not math.isfinite(value):
                raise CQVInvariantError(
                    f"feature {feature_id!r} is non-finite ({value!r}) at fold time"
                )
            low, high = entry.bounds
            if value < low - _BOUND_EPS or value > high + _BOUND_EPS:
                raise CQVInvariantError(
                    f"feature {feature_id!r} value {value!r} outside bounds "
                    f"[{low}, {high}]"
                )
            target[entry.index] = np.float32(value)


__all__: tuple[str, ...] = ("CQVAssembler",)