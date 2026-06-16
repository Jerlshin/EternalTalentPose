"""Shared composition-root carrier for both pipelines.

Owner layer: pipelines (application / composition root).
Allowed imports: ``domain``, ``ports``, ``config.schema``, stdlib (Repository
Layout §12, §16). This module declares the *base* ``RunContext`` — the
immutable carrier handed to stages — that the online and offline composition
roots specialize. It instantiates no adapters itself; the concrete pipelines
bind adapters to ports and pass them in.

The base records the reproducibility provenance every run report needs
(``code_version``, ``config_hash``, ``seed``, ``as_of``) so that the
``reproducible`` block (Ports §13) is byte-stable across rebuilds. Wall-clock
identity (``run_id``, ``started_at``) is *audit* data and explicitly lives off
this object — it must never enter a reproducibility hash.

This is a pure dataclass carrier: frozen, slotted, no IO, no clock reads, no
unseeded RNG. Determinism is configuration owned in ``config.determinism`` and
asserted by the composition root before a context is built; by the time a
``RunContext`` exists the thread pins and seeds are already fixed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redstack.config.schema import RedstackConfig

__all__: tuple[str, ...] = (
    "RunContext",
    "canonical_config_hash",
)


def canonical_config_hash(config: RedstackConfig) -> str:
    """Return a deterministic sha256 over the validated configuration.

    The hash is taken over Pydantic's canonical JSON serialization with keys
    sorted, so it is stable across process restarts and insertion order. Enum
    values serialize by value (``StrEnum``), matching the determinism policy.
    This feeds the run report's ``reproducible.config_hash`` (Ports §13) and the
    offline staleness key's ``config_slice`` (Offline Pipeline Part 11).
    """
    payload = config.model_dump_json(by_alias=False)
    sorted_payload = _canonicalize_json(payload)
    return hashlib.sha256(sorted_payload.encode("utf-8")).hexdigest()


def _canonicalize_json(payload: str) -> str:
    """Re-serialize a JSON string with sorted keys and no incidental whitespace."""
    import json

    parsed: object = json.loads(payload)
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunContext:
    """Immutable base carrier: resolved config + reproducibility provenance.

    Subclasses (``OfflinePipelineContext`` and the online ``OnlineRunContext``)
    add their bound ports and output roots. The base owns only what both share
    and what the run report's ``reproducible`` block requires.

    Attributes:
        config: The fully-composed, validated typed configuration. Raw ``dict``
            config never crosses a module boundary (Architecture §20).
        seed: The run seed sourced from ``config.determinism.seed`` (offline may
            also override from ``config.offline.seed``); the single root of every
            seeded substream. Online ranking has no legitimate RNG.
        as_of: The injected reference date. No module reads the OS clock for
            logic; recency math takes this value (Repository Layout §0).
        code_version: The build's code provenance (e.g. ``git describe``),
            recorded into the report so a submission is traceable to its source.
        config_hash: Deterministic sha256 of ``config`` (see
            :func:`canonical_config_hash`), pinned at construction.
    """

    config: RedstackConfig
    seed: int
    as_of: date
    code_version: str
    config_hash: str