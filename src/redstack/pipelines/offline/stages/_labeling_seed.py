"""Gold-label review seed DTOs — the offline workspace output O8 consumes.

The labeling workspace (Offline Pipeline Part 7) is an offline tool (notebook /
Streamlit), not part of the pipeline code. Its committed human review tags arrive
into O8 as this frozen seed (loaded by the composition root from
``paths.golden_labels_path``), exactly as the lexicon/anchor authoring seeds reach
O4/O6. O8 then performs the deterministic, machine-side work: active-learning
stratification and the seeded, leakage-free train/val split.

A ``ReviewTag`` is one reviewer's committed judgement for one candidate: the
fit-tier (``RelevanceTier`` 0–4; honeypots → 0), the human reference reasoning
(seeds O16), the reviewer id + timestamp (audit), and the stratification keys the
workspace recorded (archetype, honeypot-suspect, borderline). These are *human
facts* — fixed once committed (Part 7 determinism).
"""

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