
from __future__ import annotations

from datetime import date
from typing import Final, final

from pydantic import BaseModel, ConfigDict

from redstack.config.schema import BehavioralPolicy
from redstack.domain.candidate.behavioral import BehavioralProfile
from redstack.domain.enums import SignalAvailability
from redstack.domain.ids import Multiplier

# The five normalized families this engine reads off the profile, in fixed
# reduction order (determinism: weighted sum follows this order).
_FAMILIES: Final[tuple[str, ...]] = (
    "availability",
    "responsiveness",
    "engagement",
    "reliability",
    "verification",
)


@final
class BehavioralEngine(BaseModel):
    """Stateless, pure behavioral multiplier engine."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    policy: BehavioralPolicy

    # ------------------------------------------------------------------ public
    def multiplier(self, profile: BehavioralProfile, *, as_of: date) -> Multiplier:
        """Compose the bounded behavioral multiplier in ``[m_min, m_max]``.

        ``as_of`` is accepted for signature parity with the recency-aware policy;
        the profile's families are already ``as_of``-relative, so no clock is read.
        """
        del as_of  # no wall-clock; recency already baked into the profile

        values = self._family_values(profile)
        availabilities = self._family_availability(profile)

        weighted_sum = 0.0
        weight_total = 0.0
        for family in _FAMILIES:
            if availabilities[family] is SignalAvailability.UNKNOWN:
                continue  # drop UNKNOWN; redistribute by renormalizing the divisor
            weight = self.policy.family_weights.get(family, 0.0)
            if weight <= 0.0:
                continue
            weighted_sum += weight * values[family]
            weight_total += weight

        # All-UNKNOWN (or all-zero-weight) → neutral base (no information).
        base = (
            self.policy.unknown_neutral_base
            if weight_total <= 0.0
            else weighted_sum / weight_total
        )
        return self._to_multiplier(base)

    # --------------------------------------------------------------- internals
    @staticmethod
    def _family_values(profile: BehavioralProfile) -> dict[str, float]:
        return {
            "availability": float(profile.availability),
            "responsiveness": float(profile.responsiveness),
            "engagement": float(profile.engagement),
            "reliability": float(profile.reliability),
            "verification": float(profile.verification),
        }

    @staticmethod
    def _family_availability(
        profile: BehavioralProfile,
    ) -> dict[str, SignalAvailability]:
        return {
            "availability": profile.availability_status,
            "responsiveness": profile.responsiveness_status,
            "engagement": profile.engagement_status,
            "reliability": profile.reliability_status,
            "verification": profile.verification_status,
        }

    def _to_multiplier(self, base: float) -> Multiplier:
        """Affine map of a ``[0, 1]`` base onto ``[m_min, m_max]`` (monotone, bounded)."""
        clamped = 0.0 if base <= 0.0 else 1.0 if base >= 1.0 else base
        span = self.policy.m_max - self.policy.m_min
        value = self.policy.m_min + span * clamped
        # Defensive clamp into declared bounds (never create relevance).
        bounded = (
            self.policy.m_min
            if value < self.policy.m_min
            else self.policy.m_max
            if value > self.policy.m_max
            else value
        )
        return Multiplier(bounded)


__all__: tuple[str, ...] = ("BehavioralEngine",)