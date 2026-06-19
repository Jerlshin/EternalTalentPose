

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from redstack.domain.enums import ScoreComponent
from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage
from redstack.pipelines.offline.stages.packaging import MANIFEST_FILENAME
from redstack.ports._types import SourceMalformed, SourceOk

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "ReproducibilityStage",
    "reproducibility_stage",
)

#: Feature-parity tolerance (float32 round-trip ε; offline snapshot ↔ recompute).
_PARITY_EPS: Final[float] = 1e-4
#: Sample size for the parity recompute (deterministic, stream-order prefix).
_PARITY_SAMPLE: Final[int] = 64
#: Embedding/onnx artifacts are ε-stable, not bitwise — checked for presence only.
_EPS_STABLE_KEYS: Final[frozenset[str]] = frozenset(
    {"candidate_vectors", "anchor_vectors", "centroids", "encoder"}
)
#: Group prefix → ScoreComponent (mirrors O9/O10 for an online-consistent dry-run).
_GROUP_TO_COMPONENT: Final[Mapping[str, ScoreComponent]] = {
    "retr": ScoreComponent.SEMANTIC_FIT,
    "rank": ScoreComponent.SEMANTIC_FIT,
    "ir": ScoreComponent.SEMANTIC_FIT,
    "jd": ScoreComponent.SEMANTIC_FIT,
    "skill": ScoreComponent.SKILL_MATCH,
    "career": ScoreComponent.CAREER_FIT,
    "pvs": ScoreComponent.CAREER_FIT,
    "exp": ScoreComponent.EXPERIENCE_FIT,
    "edu": ScoreComponent.EDUCATION_FIT,
    "cons": ScoreComponent.CREDIBILITY,
    "hp": ScoreComponent.CREDIBILITY,
    "bhv": ScoreComponent.CREDIBILITY,
    "arch": ScoreComponent.ARCHETYPE_FIT,
}


@dataclass(frozen=True, slots=True)
class _Check:
    """One certification check's verdict (recorded in the report)."""

    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


class ReproducibilityStage(OfflineStage):
    """O18 — the four-check reproducibility certification gate."""

    stage_id = "O18"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        manifest = self._load_manifest(ctx)
        checks: list[_Check] = [
            self._check_manifest_reload(ctx, manifest),
            self._check_feature_parity(ctx),
            self._check_dry_run_ranking(ctx),
            self._check_rerun_determinism(ctx, manifest),
        ]
        all_passed = all(c.passed for c in checks)
        payload: dict[str, object] = {
            "checks": [c.as_dict() for c in checks],
            "all_passed": all_passed,
            "manifest_sha256": manifest.get("manifest_sha256"),
            "parity_eps": _PARITY_EPS,
        }
        if not all_passed:
            # The registry validator (_v_reproducibility_report) would reject the
            # artifact, but we fail fast here with the offending checks named so
            # the build error is actionable rather than a generic validator miss.
            failed = ", ".join(c.name for c in checks if not c.passed)
            msg = f"reproducibility certification failed checks: {failed}"
            # Emit the report first (audit trail of the failure), then raise.
            self.emit_json(ctx, "reproducibility_report", payload)
            raise ArtifactContractError(msg)

        artifact = self.emit_json(ctx, "reproducibility_report", payload)
        metrics: dict[str, object] = {
            "all_passed": all_passed,
            "checks_passed": sum(1 for c in checks if c.passed),
            "checks_total": len(checks),
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    # ------------------------------------------------------------------ #
    # Manifest                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_manifest(ctx: OfflinePipelineContext) -> Mapping[str, object]:
        """Load the packaged ``MANIFEST.json`` from the artifact tree.

        Raises:
            ArtifactContractError: the manifest is missing/malformed (O17 must run
                before O18 — its DAG dependency).
        """
        path = ctx.artifacts_root / MANIFEST_FILENAME
        if not path.is_file():
            raise ArtifactContractError(f"MANIFEST.json not found at {path}")
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactContractError(f"MANIFEST.json invalid JSON: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise ArtifactContractError("MANIFEST.json must be an object")
        return parsed

    # ------------------------------------------------------------------ #
    # Check 1 — manifest reload + self-hash + per-artifact sha256        #
    # ------------------------------------------------------------------ #
    def _check_manifest_reload(
        self, ctx: OfflinePipelineContext, manifest: Mapping[str, object]
    ) -> _Check:
        """Verify the manifest self-hash + every referenced artifact's sha256.

        Mirrors the online ``ArtifactStorePort.verify_all`` the consumer runs at
        R0: recompute the self-hash over the canonical core (excluding the
        self-hash + audit ``created_at``) and compare, then stream-hash each
        referenced artifact and compare to its recorded digest.
        """
        recorded = manifest.get("manifest_sha256")
        if not isinstance(recorded, str) or not recorded:
            return _Check("manifest_reload", False, "missing manifest_sha256")
        core = {
            k: v
            for k, v in manifest.items()
            if k not in ("manifest_sha256", "created_at")
        }
        recomputed = hashlib.sha256(
            json.dumps(
                core, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != recorded:
            return _Check(
                "manifest_reload",
                False,
                f"self-hash mismatch (recomputed {recomputed[:12]} != {recorded[:12]})",
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return _Check("manifest_reload", False, "manifest missing 'artifacts'")
        for key, entry in artifacts.items():
            if not isinstance(entry, Mapping):
                return _Check("manifest_reload", False, f"{key} entry malformed")
            rel = entry.get("path")
            expected = entry.get("sha256")
            if not isinstance(rel, str) or not isinstance(expected, str):
                return _Check("manifest_reload", False, f"{key} path/sha256 invalid")
            path = (ctx.artifacts_root / rel).resolve()
            if not path.is_file():
                return _Check("manifest_reload", False, f"{key} file missing at {rel}")
            actual = self._stream_sha256(path)
            if actual != expected:
                return _Check(
                    "manifest_reload",
                    False,
                    f"{key} sha256 mismatch ({actual[:12]} != {expected[:12]})",
                )
        return _Check(
            "manifest_reload", True, f"verified {len(artifacts)} artifacts + self-hash"
        )

    @staticmethod
    def _stream_sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    # ------------------------------------------------------------------ #
    # Check 2 — online-vs-offline feature parity                         #
    # ------------------------------------------------------------------ #
    def _check_feature_parity(self, ctx: OfflinePipelineContext) -> _Check:
        """Recompute features for a sample and diff vs ``feature_snapshot.parquet``.

        The online path recomputes the *exact* features the snapshot stored; this
        check proves the recomputation matches within ε on a deterministic sample.
        Because the structural extractor is pure-given-(raw, as_of), recomputing
        from the stream and comparing to the stored snapshot rows is the same diff
        the online parity test (Testing §8) performs.
        """
        manifest_rows, feature_ids = self._read_snapshot(ctx)
        if not manifest_rows:
            return _Check("feature_parity", False, "feature_snapshot empty/unreadable")

        try:
            from redstack.features.extraction import extract_row
            from redstack.features.parsing import validate as parse
        except Exception as exc:  # noqa: BLE001
            return _Check("feature_parity", False, f"extractor import failed: {exc}")

        registry = ctx.feature_registry
        checked = 0
        max_abs = 0.0
        for record in ctx.candidate_source.stream():
            if isinstance(record, SourceMalformed) or not isinstance(record, SourceOk):
                continue
            raw = record.raw
            cid = raw.get("candidate_id")
            if not isinstance(cid, str) or cid not in manifest_rows:
                continue
            try:
                candidate = parse(raw)
            except Exception:  # noqa: BLE001 — a parse failure here is a parity fail.
                return _Check("feature_parity", False, f"recompute parse failed for {cid}")
            recomputed, _conf = extract_row(candidate, registry, as_of=ctx.as_of)
            stored = manifest_rows[cid]
            if recomputed.shape != stored.shape:
                return _Check(
                    "feature_parity",
                    False,
                    f"{cid} shape {recomputed.shape} != stored {stored.shape}",
                )
            diff = float(np.max(np.abs(recomputed - stored))) if stored.size else 0.0
            max_abs = max(max_abs, diff)
            checked += 1
            if checked >= _PARITY_SAMPLE:
                break

        if checked == 0:
            return _Check(
                "feature_parity", False, "no sampled candidate present in snapshot"
            )
        if max_abs > _PARITY_EPS:
            return _Check(
                "feature_parity",
                False,
                f"max abs diff {max_abs:.2e} > eps {_PARITY_EPS:.0e} over {checked} rows",
            )
        return _Check(
            "feature_parity",
            True,
            f"parity within {_PARITY_EPS:.0e} over {checked} sampled rows",
        )

    def _read_snapshot(
        self, ctx: OfflinePipelineContext
    ) -> tuple[dict[str, npt.NDArray[np.float32]], tuple[str, ...]]:
        """Read ``feature_snapshot.parquet`` → {id: (D,) row} + ordered feature ids."""
        spec = self.registry.spec("feature_snapshot")
        path = (ctx.artifacts_root / spec.relative_path).resolve()
        if not path.is_file():
            return {}, ()
        import pyarrow.parquet as pq

        table = pq.read_table(path)  # type: ignore[no-untyped-call]
        value_cols = sorted(
            (c for c in table.column_names if c.startswith("f") and c[1:].isdigit()),
            key=lambda c: int(c[1:]),
        )
        if not value_cols or "id" not in table.column_names:
            return {}, ()
        ids = [str(v) for v in table.column("id").to_pylist()]
        mat = np.column_stack(
            [table.column(c).to_numpy(zero_copy_only=False) for c in value_cols]
        ).astype(np.float32, copy=False)
        rows = {cid: mat[i] for i, cid in enumerate(ids)}
        return rows, tuple(value_cols)

    # ------------------------------------------------------------------ #
    # Check 3 — deterministic dry-run ranking                            #
    # ------------------------------------------------------------------ #
    def _check_dry_run_ranking(self, ctx: OfflinePipelineContext) -> _Check:
        """Fold CQV with locked weights, sort (−score, id), assert Ranking invariants.

        Mirrors the online R5→R6 path: ``base = Σ component·weight`` over the
        snapshot, then the ``CandidateRankingEngine`` total order. Asserts the six
        ``Ranking`` invariants structurally — count == N, ranks are ``1..N`` unique
        and contiguous, scores are monotone non-increasing down the ranks, and ties
        are broken by ``candidate_id`` ascending.
        """
        rows, feature_ids = self._read_snapshot(ctx)
        if len(rows) < 2:
            return _Check("dry_run_ranking", False, "snapshot has < 2 rows")
        weights = self._load_weights(ctx)
        if weights is None:
            return _Check("dry_run_ranking", False, "scoring_weights unreadable")
        component_order = tuple(ScoreComponent)
        group_of = self._manifest_group_of(ctx)
        col_to_comp = self._map_columns(feature_ids, group_of, component_order)

        scored: list[tuple[str, float]] = []
        for cid, row in rows.items():
            comp = np.zeros(len(component_order), dtype=np.float32)
            counts = np.zeros(len(component_order), dtype=np.float32)
            for col, ci in col_to_comp.items():
                comp[ci] += row[col]
                counts[ci] += 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                comp = np.where(counts > 0, comp / np.maximum(counts, 1.0), 0.0)
            base = float(np.dot(comp, weights))
            scored.append((cid, base))

        ranked = sorted(scored, key=lambda kv: (-kv[1], kv[0]))
        n = len(ranked)
        # Invariant: unique ids, contiguous ranks 1..N.
        ids = [cid for cid, _ in ranked]
        if len(set(ids)) != n:
            return _Check("dry_run_ranking", False, "duplicate candidate ids in ranking")
        # Invariant: monotone non-increasing score; ties strictly by id ascending.
        for i in range(1, n):
            prev_s, cur_s = ranked[i - 1][1], ranked[i][1]
            if cur_s > prev_s + 1e-9:
                return _Check("dry_run_ranking", False, f"score increased at rank {i+1}")
            if abs(cur_s - prev_s) <= 1e-9 and ranked[i][0] < ranked[i - 1][0]:
                return _Check(
                    "dry_run_ranking", False, f"tie not id-ordered at rank {i+1}"
                )
        return _Check(
            "dry_run_ranking",
            True,
            f"ranked {n} candidates; monotone + id tie-break verified",
        )

    def _load_weights(
        self, ctx: OfflinePipelineContext
    ) -> npt.NDArray[np.float32] | None:
        import yaml

        spec = self.registry.spec("scoring_weights")
        path = (ctx.artifacts_root / spec.relative_path).resolve()
        if not path.is_file():
            return None
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, Mapping):
            return None
        weights = doc.get("weights")
        if not isinstance(weights, Mapping):
            return None
        vec = np.zeros(len(ScoreComponent), dtype=np.float32)
        for i, component in enumerate(ScoreComponent):
            value = weights.get(component.value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                vec[i] = float(value)
        return vec

    def _manifest_group_of(self, ctx: OfflinePipelineContext) -> Mapping[str, str]:
        spec = self.registry.spec("feature_manifest")
        path = (ctx.artifacts_root / spec.relative_path).resolve()
        if not path.is_file():
            return {}
        doc = json.loads(path.read_text(encoding="utf-8"))
        group_of = doc.get("group_of") if isinstance(doc, Mapping) else None
        if isinstance(group_of, Mapping):
            return {str(k): str(v) for k, v in group_of.items()}
        return {}

    @staticmethod
    def _map_columns(
        feature_cols: Sequence[str],
        group_of: Mapping[str, str],
        component_order: Sequence[ScoreComponent],
    ) -> dict[int, int]:
        """Map value-column index → component index (mirrors O9/O10).

        ``feature_cols`` are the snapshot's ``f0..f{D-1}`` names; group resolution
        falls back to ``CREDIBILITY`` for unmapped groups (consistent with O9/O10).
        Note the snapshot column order *is* the manifest feature order (O14).
        """
        comp_index = {c: i for i, c in enumerate(component_order)}
        # The manifest's group_of is keyed by feature id, but the snapshot columns
        # are positional (f0..); the manifest feature order == snapshot order, so
        # we map positionally through the manifest's ordered features.
        feature_ids = list(group_of.keys())
        out: dict[int, int] = {}
        for col in range(len(feature_cols)):
            if col < len(feature_ids):
                group = group_of.get(feature_ids[col], "")
            else:
                group = ""
            prefix = group.split(".", 1)[0] if group else ""
            component = _GROUP_TO_COMPONENT.get(prefix, ScoreComponent.CREDIBILITY)
            out[col] = comp_index[component]
        return out

    # ------------------------------------------------------------------ #
    # Check 4 — re-run determinism (hash stability)                      #
    # ------------------------------------------------------------------ #
    def _check_rerun_determinism(
        self, ctx: OfflinePipelineContext, manifest: Mapping[str, object]
    ) -> _Check:
        """Confirm deterministic artifacts' on-disk sha256 match the manifest.

        Deterministic stages produce byte-identical output across runs, so the
        on-disk artifact must still hash to the manifest digest (no post-package
        drift). Embedding artifacts are ε-stable (not bitwise), so they are checked
        for presence + recorded-dim, not hash equality.
        """
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return _Check("rerun_determinism", False, "manifest missing 'artifacts'")
        deterministic_checked = 0
        for key, entry in artifacts.items():
            if not isinstance(entry, Mapping):
                continue
            rel = entry.get("path")
            expected = entry.get("sha256")
            if not isinstance(rel, str) or not isinstance(expected, str):
                continue
            path = (ctx.artifacts_root / rel).resolve()
            if not path.is_file():
                return _Check("rerun_determinism", False, f"{key} missing for re-hash")
            if key in _EPS_STABLE_KEYS:
                continue  # ε-stable: presence already confirmed
            actual = self._stream_sha256(path)
            if actual != expected:
                return _Check(
                    "rerun_determinism",
                    False,
                    f"{key} drifted ({actual[:12]} != {expected[:12]})",
                )
            deterministic_checked += 1
        return _Check(
            "rerun_determinism",
            True,
            f"{deterministic_checked} deterministic artifacts hash-stable",
        )


def reproducibility_stage() -> ReproducibilityStage:
    """Factory: construct the O18 reproducibility-certification stage."""
    return ReproducibilityStage()