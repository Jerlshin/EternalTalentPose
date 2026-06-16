"""O8 — Candidate Labeling Workspace (stratification + leakage-free split).

Owner stage: O8 (Offline Pipeline Part 7). The human-in-the-loop review is an
offline tool whose committed tags arrive as a ``GoldLabelSeed`` (injected at
construction). O8 performs the deterministic machine-side work: the
**active-learning stratification** and the **seeded, leakage-free** train/val
split that O9/O15 calibrate on (Part 7 outputs ``GoldLabelDataset`` +
``CalibrationDataset``).

Algorithm (deps O7, O14):

1. Materialize the ``GoldLabelDataset`` from the committed tags: candidate_id →
   tier (honeypots forced to 0), reasoning, reviewer, cited features (the O16
   seed), and the stratification keys the workspace recorded.
2. **Stratify** by ``(archetype_id, honeypot_suspect, borderline)`` — the
   active-learning cells that put the most decision-relevant candidates (near the
   top-100 cut, near eligibility boundaries) in their own strata.
3. **Split per stratum, seeded** (``OfflineEntropy.numpy_generator("calibration_split")``):
   a fixed validation fraction of *each* stratum goes to val, the rest to train,
   so every cell is represented in both blocks and the split is reproducible.
   The split is over candidate ids — each id lands in exactly one block, so the
   ``_v_calibration_split`` disjointness (no leakage) holds by construction.

Outputs (delegated to the bound store):

* ``gold_labels.json`` — ``{labels: {id: {tier, reasoning, reviewer, archetype_id,
  cited_features, behavioral_composite?}}}``; ``_v_gold_labels`` (tiers ∈ 0..4).
* ``calibration_split.json`` — ``{train_ids, val_ids, strata}``;
  ``_v_calibration_split`` (disjoint train/val).

Determinism (Part 7): tags are human facts (fixed); the split is seeded and the
per-stratum shuffle uses the entropy port's labeled substream, so identical seed
⇒ identical split. Ids within each block are emitted sorted for byte-stability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage
from redstack.pipelines.offline.stages._labeling_seed import GoldLabelSeed, ReviewTag

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "LabelingStage",
    "labeling_stage",
)

#: Fraction of *each stratum* assigned to validation (the rest to train).
_VAL_FRACTION: Final[float] = 0.25
#: Minimum total labeled candidates required to form a usable calibration set.
_MIN_LABELS: Final[int] = 8


@dataclass(slots=True)
class _Stratum:
    """One active-learning stratum: a stratification key + its member ids."""

    key: tuple[int, bool, bool]
    ids: list[str] = field(default_factory=list)


class LabelingStage(OfflineStage):
    """O8 — materialize gold labels + the seeded, leakage-free calibration split."""

    stage_id = "O8"
    stage_version = "1.0"

    def __init__(self, seed: GoldLabelSeed, registry: object | None = None) -> None:
        """Construct with the injected committed human review tags.

        Args:
            seed: the workspace's committed ``GoldLabelSeed`` (Part 7 output).
            registry: optional artifact registry override.
        """
        if registry is None:
            super().__init__()
        else:
            from redstack.pipelines.offline.registry import OfflineArtifactRegistry

            assert isinstance(registry, OfflineArtifactRegistry)
            super().__init__(registry)
        if len(seed.candidate_ids) < _MIN_LABELS:
            msg = (
                f"gold label seed has {len(seed.candidate_ids)} labels; "
                f"need ≥ {_MIN_LABELS} for a usable calibration set"
            )
            raise ArtifactContractError(msg)
        self._seed = seed

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        tags = self._seed.by_id()
        labels = self._build_labels(tags)
        strata = self._stratify(tags)
        train_ids, val_ids = self._split(ctx, strata)

        gold_artifact = self.emit_json(ctx, "gold_labels", {"labels": labels})
        split_artifact = self.emit_json(
            ctx,
            "calibration_split",
            {
                "train_ids": sorted(train_ids),
                "val_ids": sorted(val_ids),
                "val_fraction": _VAL_FRACTION,
                "strata": [
                    {"key": list(s.key), "size": len(s.ids)} for s in strata
                ],
            },
        )
        metrics: dict[str, object] = {
            "labeled": len(labels),
            "train": len(train_ids),
            "val": len(val_ids),
            "strata": len(strata),
            "honeypot_zeroed": sum(
                1 for t in tags.values() if t.is_honeypot_suspect
            ),
        }
        return StageResult(
            artifacts=(gold_artifact, split_artifact), metrics=metrics
        )

    # ------------------------------------------------------------------ #
    # Gold dataset                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_labels(tags: Mapping[str, ReviewTag]) -> dict[str, object]:
        """Materialize the GoldLabelDataset; honeypot suspects are forced to tier 0.

        Part 7: honeypots → tier 0. We honor the reviewer's tier but clamp any
        honeypot-suspect to 0 regardless of the recorded tier, so a mislabeled
        honeypot cannot leak a positive tier into calibration.
        """
        labels: dict[str, object] = {}
        for cid in sorted(tags):
            tag = tags[cid]
            tier = 0 if tag.is_honeypot_suspect else tag.tier
            labels[cid] = {
                "tier": tier,
                "reasoning": tag.reasoning,
                "reviewer": tag.reviewer,
                "archetype_id": tag.archetype_id,
                "cited_features": list(tag.cited_features),
                "is_honeypot_suspect": tag.is_honeypot_suspect,
                "is_borderline": tag.is_borderline,
            }
        return labels

    # ------------------------------------------------------------------ #
    # Stratification (active learning cells)                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stratify(tags: Mapping[str, ReviewTag]) -> list[_Stratum]:
        """Group ids into ``(archetype, honeypot_suspect, borderline)`` strata.

        These are the active-learning cells (Part 7): borderline + honeypot-suspect
        candidates near the decision boundaries get their own strata so both train
        and val see them. Strata are returned in a deterministic key order.
        """
        buckets: dict[tuple[int, bool, bool], _Stratum] = {}
        for cid in sorted(tags):
            tag = tags[cid]
            key = (
                tag.archetype_id if tag.archetype_id is not None else -1,
                tag.is_honeypot_suspect,
                tag.is_borderline,
            )
            stratum = buckets.get(key)
            if stratum is None:
                stratum = _Stratum(key=key)
                buckets[key] = stratum
            stratum.ids.append(cid)
        return [buckets[k] for k in sorted(buckets)]

    # ------------------------------------------------------------------ #
    # Seeded, leakage-free split                                         #
    # ------------------------------------------------------------------ #
    def _split(
        self, ctx: OfflinePipelineContext, strata: Sequence[_Stratum]
    ) -> tuple[list[str], list[str]]:
        """Split each stratum train/val with the seeded entropy substream.

        Each id lands in exactly one block (leakage-free by construction). Within a
        stratum the ids are deterministically shuffled by the
        ``"calibration_split"`` substream, then the first ``_VAL_FRACTION`` go to
        val. A singleton stratum contributes its sole id to train (so val never
        starves train and vice-versa for tiny cells).
        """
        generator = ctx.entropy.numpy_generator("calibration_split")
        train: list[str] = []
        val: list[str] = []
        for stratum in strata:
            ids = sorted(stratum.ids)  # stable base order before the seeded shuffle
            order = generator.permutation(len(ids))
            shuffled = [ids[i] for i in order.tolist()]
            n_val = int(len(shuffled) * _VAL_FRACTION)
            if len(shuffled) >= 2 and n_val == 0:
                n_val = 1  # ensure multi-member strata contribute to val
            val.extend(shuffled[:n_val])
            train.extend(shuffled[n_val:])
        # Disjointness is structural (each id appears once), but assert defensively.
        if set(train) & set(val):
            raise ArtifactContractError("calibration split produced overlapping ids")
        if not train or not val:
            raise ArtifactContractError(
                "calibration split degenerate (one block empty); need more strata"
            )
        return train, val


def labeling_stage(seed: GoldLabelSeed) -> LabelingStage:
    """Factory: construct the O8 labeling stage bound to the injected seed."""
    return LabelingStage(seed)