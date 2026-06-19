
from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict

from redstack.config.schema import LogisticsPolicy
from redstack.domain.candidate.logistics import LogisticsProfile
from redstack.domain.enums import LocationFit, NoticeFit
from redstack.domain.ids import Multiplier


@final
class LogisticsEngine(BaseModel):
    """Stateless, pure logistics multiplier engine."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    policy: LogisticsPolicy

    # ------------------------------------------------------------------ public
    def multiplier(self, profile: LogisticsProfile) -> Multiplier:
        """Compose the bounded logistics multiplier in ``[m_min, m_max]``.

        Combination order (fixed, deterministic): location-fit factor × notice-fit
        factor × work-mode contribution × salary-inversion dampen, then clamped
        into the policy bounds.
        """
        location_factor = self._location_factor(profile.location_fit)
        notice_factor = self._notice_factor(profile.notice_fit)

        # Work-mode fit (already a UnitScore upstream) contributes via a bounded
        # weight: a perfect work-mode match contributes ``work_mode_weight``,
        # a total mismatch contributes nothing.
        work_mode = 1.0 + self.policy.work_mode_weight * (
            float(profile.work_mode_fit) - 1.0
        )

        salary_dampen = (
            self.policy.salary_inversion_factor if profile.salary.is_inverted else 1.0
        )

        raw = location_factor * notice_factor * work_mode * salary_dampen
        return self._clamp(raw)

    # --------------------------------------------------------------- internals
    def _location_factor(self, fit: LocationFit) -> float:
        return self.policy.location_fit_factor.get(fit, self.policy.location_default_factor)

    def _notice_factor(self, fit: NoticeFit) -> float:
        return self.policy.notice_fit_factor.get(fit, self.policy.notice_default_factor)

    def _clamp(self, value: float) -> Multiplier:
        bounded = (
            self.policy.m_min
            if value < self.policy.m_min
            else self.policy.m_max
            if value > self.policy.m_max
            else value
        )
        return Multiplier(bounded)


__all__: tuple[str, ...] = ("LogisticsEngine",)