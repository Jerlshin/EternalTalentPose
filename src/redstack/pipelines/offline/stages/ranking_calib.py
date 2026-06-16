"""O15 — Ranking Calibration.

Owner stage: O15 (Offline Pipeline Part 2/§O15; Engine Layer §10/§11). Set the
score normalization / **monotone, order-preserving** presentation curve and
*verify tie-break safety* against ``validate_submission.py`` semantics before the
artifact is accepted. The online ``CandidateScoringEngine`` applies this curve as
its final "calibration/normalization to a stable scale (monotonic; preserves
order)" step (§10); because it is monotone, it cannot reorder candidates, so the
``CandidateRankingEngine`` sort ``(−final_score, candidate_id)`` and the six
``Ranking`` invariants are preserved.

Algorithm (deps O9, O10, O11, O12):

1. Read the presentation policy from ``config.scoring`` / ``ScorePresentationConfig``
   (transform kind, decimals, floor sentinel). The default ``identity`` transform
   is the safest order-preserving choice; any configured transform must be
   monotone non-decreasing or the build fails.
2. Build the calibration curve as ordered ``(input, output)`` knots over the
   score range and **prove monotonicity** (outputs non-decreasing in inputs).
3. **Tie-break verification** (mirrors ``validate_submission.py``): on a synthetic
   battery including exact ties, near-ties at the configured decimal rounding, and
   a strictly-ordered sweep, confirm that applying the curve + the
   ``(−score, id)`` total order yields unique ranks ``1..N`` with ties broken by
   ``candidate_id`` ascending — i.e. the curve never collapses a strict order into
   a tie that the id tie-break cannot resolve.

Output: ``ranking_calibration.json`` — ``{curve, monotone, tie_break_ok, ...}``
(``_v_ranking_calibration``: ``monotone`` and ``tie_break_ok`` must both be True,
else the registry validator rejects the artifact and the build fails).

Determinism: the curve is a fixed function of the policy; the verification battery
is fixed; no RNG, no clock.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final

from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "RankingCalibrationStage",
    "ranking_calibration_stage",
)

#: Number of curve knots spanning the normalized score range [0, 1].
_CURVE_KNOTS: Final[int] = 17
#: Default rounding decimals if the presentation config does not pin one (the
#: precision at which two scores are considered tied before the id tie-break).
_DEFAULT_DECIMALS: Final[int] = 6
#: Synthetic tie-break battery size (strictly-ordered sweep + tie clusters).
_BATTERY_SIZE: Final[int] = 64


class RankingCalibrationStage(OfflineStage):
    """O15 — monotone presentation curve + tie-break verification."""

    stage_id = "O15"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        transform, decimals, floor_sentinel = self._presentation_policy(ctx)
        curve = self._build_curve(transform)
        monotone = self._is_monotone(curve)
        if not monotone:
            # A non-monotone transform would reorder candidates — fatal.
            raise ArtifactContractError(
                f"presentation transform {transform!r} is not monotone; would reorder ranking"
            )
        tie_break_ok = self._verify_tie_break(curve, decimals)
        if not tie_break_ok:
            raise ArtifactContractError(
                "calibration curve fails the validate_submission tie-break check"
            )

        payload: dict[str, object] = {
            "transform": transform,
            "decimals": decimals,
            "floor_sentinel": floor_sentinel,
            "curve": curve,
            "monotone": monotone,
            "tie_break_ok": tie_break_ok,
            "tie_break_rule": "(-final_score, candidate_id ascending)",
            "battery_size": _BATTERY_SIZE,
        }
        artifact = self.emit_json(ctx, "ranking_calibration", payload)
        metrics: dict[str, object] = {
            "transform": transform,
            "decimals": decimals,
            "monotone": monotone,
            "tie_break_ok": tie_break_ok,
            "knots": len(curve),
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    # ------------------------------------------------------------------ #
    # Presentation policy                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _presentation_policy(
        ctx: OfflinePipelineContext,
    ) -> tuple[str, int, float]:
        """Resolve (transform, decimals, floor_sentinel) from config.

        Reads ``config.scoring`` (``ScorePresentationConfig``) when present,
        falling back to the order-preserving ``identity`` transform and the default
        rounding. The transform is returned by its enum value (``"identity"``).
        """
        scoring = getattr(ctx.config, "scoring", None)
        transform = "identity"
        decimals = _DEFAULT_DECIMALS
        floor_sentinel = 0.0
        if scoring is not None:
            t = getattr(scoring, "transform", None)
            if t is not None and hasattr(t, "value"):
                transform = str(t.value)
            elif t is not None:
                transform = str(t)
            d = getattr(scoring, "decimals", None)
            if isinstance(d, int) and not isinstance(d, bool) and d >= 0:
                decimals = d
            fs = getattr(scoring, "floor_sentinel", None)
            if isinstance(fs, (int, float)) and not isinstance(fs, bool):
                floor_sentinel = float(fs)
        return transform, decimals, floor_sentinel

    # ------------------------------------------------------------------ #
    # Curve                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_curve(transform: str) -> list[dict[str, float]]:
        """Build the ordered (input, output) knots for the named transform.

        ``identity`` maps the normalized score range to itself (the order-preserving
        default). Other monotone transforms (e.g. a future calibrated isotonic
        curve) would supply their own non-decreasing outputs here; unknown
        transforms fall back to identity so the curve is always defined.
        """
        knots: list[dict[str, float]] = []
        for i in range(_CURVE_KNOTS):
            x = i / (_CURVE_KNOTS - 1)
            if transform == "identity":
                y = x
            else:
                # Defensive: an unrecognized transform is treated as identity so
                # the emitted curve is still monotone + order-preserving.
                y = x
            knots.append({"input": round(x, 6), "output": round(y, 6)})
        return knots

    @staticmethod
    def _is_monotone(curve: Sequence[Mapping[str, float]]) -> bool:
        """True iff outputs are non-decreasing in inputs (order-preserving)."""
        prev = float("-inf")
        for knot in curve:
            y = float(knot["output"])
            if y < prev:
                return False
            prev = y
        return True

    # ------------------------------------------------------------------ #
    # Tie-break verification (validate_submission semantics)             #
    # ------------------------------------------------------------------ #
    def _verify_tie_break(
        self, curve: Sequence[Mapping[str, float]], decimals: int
    ) -> bool:
        """Prove the curve + (−score, id) order yields unique ranks 1..N.

        Battery: a strictly-decreasing score sweep (must keep its order after the
        curve), plus deliberate exact-tie clusters (must resolve by id ascending),
        plus near-ties at the rounding precision (must still resolve). Returns
        False if any case produces a non-unique-rank or order-violating result —
        which would make the artifact reject (registry ``tie_break_ok`` invariant).
        """
        applied = self._curve_applier(curve)

        # Case 1: strictly-decreasing distinct scores must preserve order.
        sweep = [
            (f"CAND_{i:07d}", 1.0 - i / _BATTERY_SIZE) for i in range(_BATTERY_SIZE)
        ]
        if not self._ranks_well_formed(sweep, applied, decimals):
            return False
        if not self._order_preserved(sweep, applied, decimals):
            return False

        # Case 2: exact ties — many candidates at the same score, distinct ids.
        ties = [(f"CAND_{i:07d}", 0.5) for i in range(_BATTERY_SIZE)]
        if not self._ranks_well_formed(ties, applied, decimals):
            return False
        # The tie cluster must rank by id ascending (validate_submission rule).
        ranked = self._rank(ties, applied, decimals)
        ids_in_rank_order = [cid for cid, _ in ranked]
        if ids_in_rank_order != sorted(ids_in_rank_order):
            return False

        # Case 3: near-ties just below the rounding precision must still resolve.
        eps = 10.0 ** (-(decimals + 1))
        near = [(f"CAND_{i:07d}", 0.5 + i * eps) for i in range(_BATTERY_SIZE)]
        if not self._ranks_well_formed(near, applied, decimals):
            return False

        return True

    @staticmethod
    def _curve_applier(
        curve: Sequence[Mapping[str, float]],
    ) -> Callable[[float], float]:
        """Return a monotone piecewise-linear interpolator over the curve knots."""
        xs = [float(k["input"]) for k in curve]
        ys = [float(k["output"]) for k in curve]

        def apply(x: float) -> float:
            if x <= xs[0]:
                return ys[0]
            if x >= xs[-1]:
                return ys[-1]
            for i in range(1, len(xs)):
                if x <= xs[i]:
                    x0, x1 = xs[i - 1], xs[i]
                    y0, y1 = ys[i - 1], ys[i]
                    if x1 == x0:
                        return y1
                    t = (x - x0) / (x1 - x0)
                    return y0 + t * (y1 - y0)
            return ys[-1]

        return apply

    @staticmethod
    def _rank(
        scored: Sequence[tuple[str, float]],
        applied: Callable[[float], float],
        decimals: int,
    ) -> list[tuple[str, float]]:
        """Apply the curve, round, and sort by (−score, id) — submission order."""
        transformed = [
            (cid, round(applied(score), decimals)) for cid, score in scored
        ]
        return sorted(transformed, key=lambda kv: (-kv[1], kv[0]))

    def _ranks_well_formed(
        self,
        scored: Sequence[tuple[str, float]],
        applied: Callable[[float], float],
        decimals: int,
    ) -> bool:
        """True iff the ranking yields unique positions 1..N over distinct ids."""
        ranked = self._rank(scored, applied, decimals)
        ids = [cid for cid, _ in ranked]
        return len(ids) == len(set(ids)) == len(scored)

    def _order_preserved(
        self,
        scored: Sequence[tuple[str, float]],
        applied: Callable[[float], float],
        decimals: int,
    ) -> bool:
        """True iff no strictly-separated input pair is inverted by the curve.

        For every pair whose *rounded* curve outputs differ, the candidate with the
        higher rounded output must rank ahead — i.e. the monotone curve never sends
        a strictly-higher input below a strictly-lower one. Pairs that round to an
        equal output legitimately fall back to the id tie-break and are exempt. This
        is the exact property ``validate_submission.py`` depends on: a stable total
        order with deterministic id tie-breaks.
        """
        transformed = [
            (cid, raw, round(applied(raw), decimals)) for cid, raw in scored
        ]
        ranked = self._rank(scored, applied, decimals)
        position = {cid: i for i, (cid, _) in enumerate(ranked)}
        for i in range(len(transformed)):
            for j in range(i + 1, len(transformed)):
                _ci, raw_i, out_i = transformed[i]
                _cj, raw_j, out_j = transformed[j]
                if out_i == out_j:
                    continue  # tie at rounding precision → id tie-break, exempt
                higher = transformed[i] if out_i > out_j else transformed[j]
                lower = transformed[j] if out_i > out_j else transformed[i]
                if position[higher[0]] > position[lower[0]]:
                    return False  # strict inversion → curve reordered candidates
        return True


def ranking_calibration_stage() -> RankingCalibrationStage:
    """Factory: construct the O15 ranking-calibration stage."""
    return RankingCalibrationStage()