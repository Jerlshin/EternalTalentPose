
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from redstack.domain.errors import SchemaError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage
from redstack.ports._types import SourceMalformed, SourceOk

from redstack.pipelines.offline.context import OfflinePipelineContext


@dataclass(slots=True)
class _P2Quantile:
    """Single-pass P² quantile estimator (Jain & Chlamtac, 1985), O(1) memory.

    Tracks one quantile ``p`` with five markers, updating their positions as
    observations stream in. Deterministic for a fixed input order. Returns
    ``float('nan')`` only before five observations exist; the census guards that.
    """

    p: float
    _n: int = 0
    _q: list[float] = field(default_factory=list)
    _np: list[float] = field(default_factory=list)
    _dn: list[float] = field(default_factory=list)
    _npos: list[int] = field(default_factory=list)

    def update(self, x: float) -> None:
        """Incorporate one observation ``x`` into the estimator."""
        if self._n < 5:
            self._q.append(x)
            self._n += 1
            if self._n == 5:
                self._q.sort()
                self._npos = [1, 2, 3, 4, 5]
                self._np = [
                    1.0,
                    1.0 + 2.0 * self.p,
                    1.0 + 4.0 * self.p,
                    3.0 + 2.0 * self.p,
                    5.0,
                ]
                self._dn = [0.0, self.p / 2.0, self.p, (1.0 + self.p) / 2.0, 1.0]
            return

        if x < self._q[0]:
            self._q[0] = x
            k = 0
        elif x >= self._q[4]:
            self._q[4] = x
            k = 3
        else:
            k = 0
            for i in range(4):
                if self._q[i] <= x < self._q[i + 1]:
                    k = i
                    break
        for i in range(k + 1, 5):
            self._npos[i] += 1
        for i in range(5):
            self._np[i] += self._dn[i]
        self._adjust()
        self._n += 1

    def _adjust(self) -> None:
        """Move interior markers toward their desired positions (parabolic step)."""
        for i in range(1, 4):
            d = self._np[i] - self._npos[i]
            left = self._npos[i] - self._npos[i - 1]
            right = self._npos[i + 1] - self._npos[i]
            if (d >= 1 and right > 1) or (d <= -1 and left > 1):
                sign = 1 if d >= 0 else -1
                q_par = self._parabolic(i, sign)
                if self._q[i - 1] < q_par < self._q[i + 1]:
                    self._q[i] = q_par
                else:
                    self._q[i] = self._linear(i, sign)
                self._npos[i] += sign

    def _parabolic(self, i: int, sign: int) -> float:
        n_prev = self._npos[i - 1]
        n_cur = self._npos[i]
        n_next = self._npos[i + 1]
        q_prev = self._q[i - 1]
        q_cur = self._q[i]
        q_next = self._q[i + 1]
        a = sign / (n_next - n_prev)
        b = (n_cur - n_prev + sign) * (q_next - q_cur) / (n_next - n_cur)
        c = (n_next - n_cur - sign) * (q_cur - q_prev) / (n_cur - n_prev)
        return q_cur + a * (b + c)

    def _linear(self, i: int, sign: int) -> float:
        nbr = i + sign
        return self._q[i] + sign * (self._q[nbr] - self._q[i]) / (
            self._npos[nbr] - self._npos[i]
        )

    def value(self) -> float:
        """Return the current quantile estimate, or the exact value if n < 5."""
        if self._n == 0:
            return float("nan")
        if self._n < 5:
            ordered = sorted(self._q)
            idx = min(len(ordered) - 1, int(round(self.p * (len(ordered) - 1))))
            return ordered[idx]
        return self._q[2]


@dataclass(slots=True)
class _NumericProfile:
    """Streaming min / max / mean / count + P² quantiles for one numeric field."""

    count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    _sum: float = 0.0
    _p25: _P2Quantile = field(default_factory=lambda: _P2Quantile(0.25))
    _p50: _P2Quantile = field(default_factory=lambda: _P2Quantile(0.50))
    _p75: _P2Quantile = field(default_factory=lambda: _P2Quantile(0.75))
    _p95: _P2Quantile = field(default_factory=lambda: _P2Quantile(0.95))

    def update(self, x: float) -> None:
        self.count += 1
        self._sum += x
        self.minimum = min(self.minimum, x)
        self.maximum = max(self.maximum, x)
        self._p25.update(x)
        self._p50.update(x)
        self._p75.update(x)
        self._p95.update(x)

    def as_dict(self) -> dict[str, object]:
        if self.count == 0:
            return {"count": 0}
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self._sum / self.count,
            "p25": self._p25.value(),
            "p50": self._p50.value(),
            "p75": self._p75.value(),
            "p95": self._p95.value(),
        }


class CensusStage(OfflineStage):
    """O0 — stream the pool once and emit ``dataset_profile.json``."""

    stage_id = "O0"
    stage_version = "1.0"

    # Top-level profile fields whose presence is tallied for coverage.
    _COVERAGE_FIELDS: tuple[str, ...] = (
        "candidate_id",
        "profile",
        "career_history",
        "education",
        "skills",
        "redrob_signals",
    )

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        total = 0
        malformed = 0
        coverage: dict[str, int] = {f: 0 for f in self._COVERAGE_FIELDS}
        yoe = _NumericProfile()
        skill_counts = _NumericProfile()
        position_counts = _NumericProfile()
        country_freq: dict[str, int] = {}

        for record in ctx.candidate_source.stream():
            if isinstance(record, SourceMalformed):
                malformed += 1
                continue
            if not isinstance(record, SourceOk):  # exhaustive guard for the union
                continue
            total += 1
            raw = record.raw
            for field_name in self._COVERAGE_FIELDS:
                if field_name in raw and raw[field_name] is not None:
                    coverage[field_name] += 1
            self._profile_record(raw, yoe, skill_counts, position_counts, country_freq)

        profile: dict[str, object] = {
            "candidate_count": total,
            "malformed_count": malformed,
            "field_coverage": {
                name: {
                    "present": present,
                    "fraction": (present / total) if total else 0.0,
                }
                for name, present in coverage.items()
            },
            "distributions": {
                "years_of_experience": yoe.as_dict(),
                "skill_count": skill_counts.as_dict(),
                "position_count": position_counts.as_dict(),
                "country_frequency": dict(
                    sorted(country_freq.items(), key=lambda kv: (-kv[1], kv[0]))
                ),
            },
        }
        artifact = self.emit_json(ctx, "dataset_profile", profile)
        metrics: dict[str, object] = {
            "candidate_count": total,
            "malformed_count": malformed,
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    @staticmethod
    def _profile_record(
        raw: Mapping[str, object],
        yoe: _NumericProfile,
        skill_counts: _NumericProfile,
        position_counts: _NumericProfile,
        country_freq: dict[str, int],
    ) -> None:
        """Fold one raw record into the streaming aggregators (no materialization)."""
        profile_obj = raw.get("profile")
        if isinstance(profile_obj, Mapping):
            years = profile_obj.get("years_of_experience")
            if isinstance(years, (int, float)) and not isinstance(years, bool):
                yoe.update(float(years))
            country = profile_obj.get("country")
            if isinstance(country, str) and country:
                country_freq[country] = country_freq.get(country, 0) + 1

        skills = raw.get("skills")
        if isinstance(skills, (list, tuple)):
            skill_counts.update(float(len(skills)))

        history = raw.get("career_history")
        if isinstance(history, (list, tuple)):
            position_counts.update(float(len(history)))


def census_stage() -> CensusStage:
    """Factory: construct the O0 census stage bound to the frozen registry."""
    return CensusStage()