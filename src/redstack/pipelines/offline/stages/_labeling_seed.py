

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from pydantic import BaseModel, ConfigDict, Field

__all__: tuple[str, ...] = ("ReviewTag", "GoldLabelSeed")


@final
class ReviewTag(BaseModel):
    """One reviewer's committed judgement for one candidate (Part 7)."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    candidate_id: str
    tier: int = Field(ge=0, le=4)
    reasoning: str
    reviewer: str
    archetype_id: int | None = None
    is_honeypot_suspect: bool = False
    is_borderline: bool = False
    cited_features: tuple[str, ...] = ()


@final
class GoldLabelSeed(BaseModel):
    """The committed set of human review tags (the workspace's output)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tags: tuple[ReviewTag, ...] = Field(min_length=1)

    def by_id(self) -> dict[str, ReviewTag]:
        """Return the last-committed tag per candidate (id-keyed, dedup)."""
        out: dict[str, ReviewTag] = {}
        for tag in self.tags:
            out[tag.candidate_id] = tag
        return out

    @property
    def candidate_ids(self) -> Sequence[str]:
        """Distinct labeled candidate ids in committed order."""
        seen: dict[str, None] = {}
        for tag in self.tags:
            seen.setdefault(tag.candidate_id, None)
        return tuple(seen)