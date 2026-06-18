"""Offline composition root — binds adapters to ports and wires O0-O18.

Owner layer: pipelines (offline composition root).
Allowed imports: ``domain``, ``ports``, ``engines``, ``adapters``, ``config``,
``observability``, ``features``, this package (Repository Layout §12). Mirrors
``pipeline.py``'s own "one of the two places adapters are bound to ports"
charter — this module is where the CLI's ``build`` verb hands off; the CLI
itself may not import ``adapters`` directly (Repository Layout §8b).

:func:`run_offline_build` is the single entrypoint: it loads the authoring
seeds, constructs every adapter, builds the immutable
:class:`OfflinePipelineContext`, wires the O0-O18 stage callables (Offline
Pipeline Part 2), and executes the full ``plan → run → finalize`` lifecycle.

O8 (:class:`~redstack.pipelines.offline.stages.labeling.LabelingStage`) needs a
committed, human-curated :class:`GoldLabelSeed` (Offline Pipeline Part 7) —
real review tags from the labeling workspace, not something this composition
root can synthesize. Loading it is deferred to the moment O8 actually runs
(:class:`_LazyLabelingStage`) so a build can make real progress through every
stage that does not need it, and fails with a clear, specific error only when
O8 is reached and no seed is committed at ``paths.golden_labels_path``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

from redstack.adapters.candidate_jsonl import JsonlCandidateSourceAdapter
from redstack.adapters.entropy import OfflineEntropy
from redstack.adapters.st_embedder import SentenceTransformerEmbeddingAdapter
from redstack.config.loader import (
    load_eligibility_rules,
    load_jd_anchors,
    load_lexicon_seed,
)
from redstack.domain.errors import ArtifactContractError
from redstack.features.registry import FEATURE_REGISTRY
from redstack.pipelines.offline.build_artifact_store import BuildArtifactStore
from redstack.pipelines.offline.context import OfflinePipelineContext
from redstack.pipelines.offline.pipeline import OfflinePipeline, OfflinePipelineReport
from redstack.pipelines.offline.registry import (
    OFFLINE_ARTIFACT_REGISTRY,
    OfflineArtifactRegistry,
)
from redstack.pipelines.offline.runner import StageCallable, StageReceipt, StageResult
from redstack.pipelines.offline.stages._labeling_seed import GoldLabelSeed
from redstack.pipelines.offline.stages.archetype_discovery import (
    ArchetypeDiscoveryStage,
)
from redstack.pipelines.offline.stages.behavioral_calib import (
    BehavioralCalibrationStage,
)
from redstack.pipelines.offline.stages.census import CensusStage
from redstack.pipelines.offline.stages.embedding_gen import (
    AnchorEmbeddingStage,
    CandidateEmbeddingStage,
    CareerEmbeddingStage,
    EmbeddingManifestStage,
)
from redstack.pipelines.offline.stages.feature_importance import (
    FeatureImportanceStage,
)
from redstack.pipelines.offline.stages.feature_snapshot import FeatureSnapshotStage
from redstack.pipelines.offline.stages.honeypot_discovery import (
    HoneypotDiscoveryStage,
)
from redstack.pipelines.offline.stages.jd_concepts import JdConceptStage
from redstack.pipelines.offline.stages.labeling import LabelingStage
from redstack.pipelines.offline.stages.lexicon_discovery import LexiconDiscoveryStage
from redstack.pipelines.offline.stages.normalization import NormalizationStage
from redstack.pipelines.offline.stages.packaging import PackagingStage
from redstack.pipelines.offline.stages.ranking_calib import RankingCalibrationStage
from redstack.pipelines.offline.stages.reasoning_templates import (
    ReasoningTemplateStage,
)
from redstack.pipelines.offline.stages.reproducibility import ReproducibilityStage
from redstack.pipelines.offline.stages.risk_calib import RiskCalibrationStage
from redstack.pipelines.offline.stages.validation import ValidationStage
from redstack.pipelines.offline.stages.vocab_expansion import VocabExpansionStage
from redstack.pipelines.offline.stages.weight_search import WeightSearchStage

if TYPE_CHECKING:
    from redstack.config.schema import RedstackConfig

__all__: tuple[str, ...] = ("run_offline_build",)

#: Default offline encode batch size handed to the sentence-transformers adapter.
_ST_BATCH_SIZE: Final[int] = 32


class GoldLabelSeedMissingError(ArtifactContractError):
    """O8 needs the human-curated review workspace output; none is committed."""


def _load_gold_label_seed(path: Path) -> GoldLabelSeed:
    """Load the committed Offline Pipeline Part 7 review tags from ``path``.

    Raises:
        GoldLabelSeedMissingError: ``path`` does not exist. This is *not* a
            code bug — O8 is human-in-the-loop labeling; the seed is a
            workspace output authored by reviewers, never synthesized here.
    """
    if not path.is_file():
        raise GoldLabelSeedMissingError(
            f"O8 requires committed gold labels at {path} (Offline Pipeline "
            "Part 7: the human-in-the-loop labeling workspace's output) — "
            "none found. This is human-curated ground truth and cannot be "
            "generated by the pipeline; a reviewer must commit it first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GoldLabelSeed.model_validate(raw)


@final
class _LazyLabelingStage:
    """Defers O8's :class:`GoldLabelSeed` load until the stage actually runs.

    Lets ``plan() → run()`` reach every stage that does not depend on O8
    (O0-O7, O13*, O14) and fail only at O8 itself, with a clear message,
    instead of refusing to even wire the pipeline when no seed is committed.
    """

    stage_id: Final[str] = "O8"
    stage_version: Final[str] = LabelingStage.stage_version

    def __init__(
        self,
        golden_labels_path: Path,
        registry: OfflineArtifactRegistry = OFFLINE_ARTIFACT_REGISTRY,
    ) -> None:
        self._golden_labels_path = golden_labels_path
        self._registry = registry

    def __call__(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        seed = _load_gold_label_seed(self._golden_labels_path)
        stage = LabelingStage(seed=seed, registry=self._registry)
        return stage(ctx, upstream)


def _build_stages(
    *, configs_root: Path, golden_labels_path: Path
) -> dict[str, StageCallable]:
    """Wire every O0-O18 stage callable, loading the authoring seeds it needs."""
    lexicon_seed = load_lexicon_seed(configs_root)
    jd_anchors = load_jd_anchors(configs_root)
    eligibility_rules = load_eligibility_rules(configs_root)

    stages: tuple[StageCallable, ...] = (
        CensusStage(),
        NormalizationStage(),
        ValidationStage(),
        HoneypotDiscoveryStage(),
        LexiconDiscoveryStage(seed=lexicon_seed),
        VocabExpansionStage(),
        JdConceptStage(anchors=jd_anchors, eligibility=eligibility_rules),
        ArchetypeDiscoveryStage(),
        _LazyLabelingStage(golden_labels_path=golden_labels_path),
        WeightSearchStage(),
        FeatureImportanceStage(),
        BehavioralCalibrationStage(),
        RiskCalibrationStage(),
        CandidateEmbeddingStage(),
        AnchorEmbeddingStage(),
        CareerEmbeddingStage(),
        EmbeddingManifestStage(),
        FeatureSnapshotStage(),
        RankingCalibrationStage(),
        ReasoningTemplateStage(),
        PackagingStage(),
        ReproducibilityStage(),
    )
    return {stage.stage_id: stage for stage in stages}


def run_offline_build(
    config: RedstackConfig,
    *,
    configs_root: Path,
    code_version: str,
    force: tuple[str, ...] | None = None,
) -> OfflinePipelineReport:
    """Bind adapters, build the context, wire O0-O18, and execute the build.

    Args:
        config: The fully-composed, validated offline ``RedstackConfig``.
        configs_root: Path to the ``configs/`` directory (authoring seeds).
        code_version: The build's code provenance, recorded into the report.
        force: Stage ids to recompute regardless of checkpoint freshness.

    Returns:
        The terminal :class:`OfflinePipelineReport`.

    Raises:
        ValueError: ``config.offline`` is absent (wrong run mode).
        GoldLabelSeedMissingError: O8 is reached with no committed gold labels.
    """
    offline = config.offline
    if offline is None:
        msg = "run_offline_build requires a config with an 'offline' runtime block"
        raise ValueError(msg)

    candidates_path = Path(config.paths.candidates_path).resolve()
    artifacts_root = Path(config.paths.artifacts_root).resolve()
    golden_labels_path = Path(config.paths.golden_labels_path).resolve()

    candidate_source = JsonlCandidateSourceAdapter(candidates_path)
    embedding_model = SentenceTransformerEmbeddingAdapter(
        offline.st_model_id,
        revision=offline.st_model_revision,
        batch_size=_ST_BATCH_SIZE,
    )
    entropy = OfflineEntropy(seed=offline.seed, as_of=offline.as_of.date())
    artifact_store = BuildArtifactStore(artifacts_root, OFFLINE_ARTIFACT_REGISTRY)

    ctx = OfflinePipelineContext.build(
        config=config,
        candidate_source=candidate_source,
        embedding_model=embedding_model,
        artifact_store=artifact_store,
        entropy=entropy,
        feature_registry=FEATURE_REGISTRY,
        code_version=code_version,
    )

    stages = _build_stages(
        configs_root=configs_root, golden_labels_path=golden_labels_path
    )
    pipeline = OfflinePipeline(stages=stages)
    return pipeline.execute(ctx, force=force)
