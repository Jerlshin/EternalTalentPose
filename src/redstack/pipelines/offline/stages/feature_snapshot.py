"""O14 — Candidate Representation Construction (the uncalibrated feature snapshot).

Owner stage: O14 (Offline Pipeline Part 2/§O14). Build the canonical
``(N, D)`` feature-value matrix + ``(N, num_groups)`` group-confidence matrix by
running the pure feature extractors over the validated pool, aligned to the
frozen ``FeatureLayout`` carried on the context's ``FeatureRegistry``. This is
the calibration substrate O8/O9/O10 train on and the online correctness oracle
O18 diffs against — so it must be the *exact* layout the online path computes.

This Cluster-A build is **uncalibrated**: it materializes the structural feature
rows (the extractor's deterministic output) with neutral folded values, before
any weight/threshold calibration exists. Calibration stages consume this matrix;
they do not change its shape.

Outputs:

* ``feature_snapshot.parquet`` — columnar ``id`` + ``f0..f{D-1}`` values +
  ``c_<group>`` confidences. Validated by ``_v_feature_snapshot`` (no NaN,
  positive ``feature_dim``).
* ``feature_manifest.json`` — ``layout_version`` + the ordered feature list +
  group map (Feature Layer Part 6 ``FeatureManifest``). Validated by
  ``_v_feature_manifest`` and required online; its ``layout_version`` must agree
  with ``scoring_weights`` and ``FeatureLayout`` (Part 10 layout-bearing rule).

Memory discipline (Domain §Q / Feature Layer Part 7): the matrices are the *one*
sanctioned bulk allocation — a single ``(N, D)`` float32 array (≈ N·D·4 bytes;
26 MB at N=100K, D=64) filled **row-by-row from the stream**, never via a Python
list of per-candidate objects. The candidate source is consumed lazily; only the
preallocated arrays + the id index grow with N, and those are the deliverable.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from redstack.domain.errors import (
    ArtifactContractError,
    CQVInvariantError,
    SchemaError,
)
from redstack.features.extraction import extract_row
from redstack.features.parsing import parse
from redstack.domain.source import RawCandidate
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage
from redstack.ports._types import Malformed, Ok

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "FeatureSnapshotStage",
    "feature_snapshot_stage",
)

#: Initial row capacity; arrays grow geometrically if the source count is unknown.
_INITIAL_CAPACITY: int = 4096


class FeatureSnapshotStage(OfflineStage):
    """O14 — stream the validated pool into the canonical feature snapshot."""

    stage_id = "O14"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        registry = ctx.feature_registry
        dim = registry.dim
        groups = registry.groups
        num_groups = len(groups)
        if dim <= 0:
            msg = "feature layout has zero dimensionality; cannot build snapshot"
            raise ArtifactContractError(msg)

        # Preallocate the bulk arrays; grow geometrically when capacity is unknown.
        capacity = self._initial_capacity(ctx)
        values = np.zeros((capacity, dim), dtype=np.float32)
        confidence = np.zeros((capacity, num_groups), dtype=np.float32)
        ids: list[str] = []
        row = 0

        for record in ctx.candidate_source.stream():
            if isinstance(record, Malformed):
                # Malformed lines were already rejected at O2; skip defensively.
                continue
            if not isinstance(record, Ok):
                continue
            candidate = self._parse_or_skip(record.raw)
            if candidate is None:
                continue
            if row == capacity:
                capacity *= 2
                values = self._grow(values, capacity)
                confidence = self._grow(confidence, capacity)
            feature_row, group_conf = extract_row(
                candidate, registry, as_of=ctx.as_of
            )
            self._assert_row_shape(feature_row, group_conf, dim, num_groups)
            values[row, :] = feature_row
            confidence[row, :] = group_conf
            ids.append(str(candidate.candidate_id))
            row += 1

        if row == 0:
            msg = "feature snapshot produced zero rows (empty validated pool)"
            raise ArtifactContractError(msg)

        # Trim to the exact populated size (zero-copy view, then contiguous copy).
        values = np.ascontiguousarray(values[:row, :])
        confidence = np.ascontiguousarray(confidence[:row, :])
        self._assert_no_nan(values, confidence)

        snapshot = self.emit_parquet(
            ctx,
            "feature_snapshot",
            ids=ids,
            values=values,
            confidence=confidence,
            group_names=groups,
        )
        manifest = self.emit_json(
            ctx,
            "feature_manifest",
            {
                "layout_version": registry.layout_version,
                "features": list(registry.feature_ids),
                "groups": list(groups),
                "group_of": dict(sorted(registry.group_of.items())),
                "feature_dim": dim,
                "row_count": row,
            },
        )
        metrics: dict[str, object] = {
            "row_count": row,
            "feature_dim": dim,
            "group_count": num_groups,
            "layout_version": registry.layout_version,
        }
        return StageResult(artifacts=(snapshot, manifest), metrics=metrics)

    @staticmethod
    def _initial_capacity(ctx: OfflinePipelineContext) -> int:
        """Use the source's reported count when known, else a growable default."""
        known = ctx.candidate_source.count()
        if known is not None and known > 0:
            return known
        return _INITIAL_CAPACITY

    @staticmethod
    def _grow(
        array: npt.NDArray[np.float32], new_capacity: int
    ) -> npt.NDArray[np.float32]:
        """Return a larger float32 array preserving existing rows (geometric growth)."""
        grown = np.zeros((new_capacity, array.shape[1]), dtype=np.float32)
        grown[: array.shape[0], :] = array
        return grown

    @staticmethod
    def _parse_or_skip(raw: Mapping[str, object]) -> RawCandidate | None:
        """Parse a record to ``RawCandidate``; skip on a schema breach.

        O2 already validated the pool, so a breach here is defensive only. We skip
        rather than abort: the snapshot is built over the accepted set, and a late
        breach is recorded by O2's report, not re-raised mid-matrix.
        """
        try:
            return parse(raw)
        except SchemaError:
            return None

    @staticmethod
    def _assert_row_shape(
        feature_row: npt.NDArray[np.float32],
        group_conf: npt.NDArray[np.float32],
        dim: int,
        num_groups: int,
    ) -> None:
        """Assert the extractor returned correctly-shaped float32 rows.

        Raises:
            CQVInvariantError: wrong dim / wrong group width / wrong dtype.
        """
        if feature_row.shape != (dim,) or feature_row.dtype != np.float32:
            msg = (
                f"extractor returned feature row of shape {feature_row.shape} "
                f"dtype {feature_row.dtype}; expected ({dim},) float32"
            )
            raise CQVInvariantError(msg)
        if group_conf.shape != (num_groups,) or group_conf.dtype != np.float32:
            msg = (
                f"extractor returned group confidence of shape {group_conf.shape}; "
                f"expected ({num_groups},) float32"
            )
            raise CQVInvariantError(msg)

    @staticmethod
    def _assert_no_nan(
        values: npt.NDArray[np.float32], confidence: npt.NDArray[np.float32]
    ) -> None:
        """Assert the assembled matrices are wholly finite (all sentinels resolved).

        Raises:
            CQVInvariantError: any NaN/inf in the value or confidence matrix.
        """
        if not np.all(np.isfinite(values)):
            raise CQVInvariantError("feature snapshot values contain NaN/inf")
        if not np.all(np.isfinite(confidence)):
            raise CQVInvariantError("feature snapshot confidences contain NaN/inf")


def feature_snapshot_stage() -> FeatureSnapshotStage:
    """Factory: construct the O14 snapshot stage bound to the frozen registry."""
    return FeatureSnapshotStage()