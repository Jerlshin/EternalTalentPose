

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from redstack.pipelines.context import RunContext, canonical_config_hash

if TYPE_CHECKING:
    from redstack.config.schema import RedstackConfig, OfflineRuntimeConfig
    from redstack.features.registry import FeatureRegistry
    from redstack.ports.artifact_store import ArtifactStorePort
    from redstack.ports.candidate_source import CandidateSourcePort
    from redstack.ports.embedding import EmbeddingModelPort
    from redstack.ports.rng import DeterministicEntropyPort

__all__: tuple[str, ...] = ("OfflinePipelineContext",)


@dataclass(frozen=True, slots=True, kw_only=True)
class OfflinePipelineContext(RunContext):


    candidate_source: CandidateSourcePort
    embedding_model: EmbeddingModelPort
    artifact_store: ArtifactStorePort
    entropy: DeterministicEntropyPort
    feature_registry: FeatureRegistry
    artifacts_root: Path
    data_root: Path
    quarantine_root: Path
    checkpoints_root: Path
    layout_version: str

    @classmethod
    def build(
        cls,
        *,
        config: RedstackConfig,
        candidate_source: CandidateSourcePort,
        embedding_model: EmbeddingModelPort,
        artifact_store: ArtifactStorePort,
        entropy: DeterministicEntropyPort,
        feature_registry: FeatureRegistry,
        code_version: str,
    ) -> OfflinePipelineContext:
        """Resolve a build context from validated config + already-bound ports.

        Derives ``seed`` and ``as_of`` from the offline runtime block and the
        injected entropy port (the single clock/RNG seam), computes the canonical
        ``config_hash``, and resolves the output roots under ``paths``. The
        roots are resolved to absolute paths for stable, traversal-safe IO by the
        adapters; this constructor performs no IO itself.

        Raises:
            ValueError: if invoked on a config whose ``run_mode`` is not offline
                (the offline block must be present — guaranteed by
                ``RedstackConfig`` mode-consistency, re-asserted here for the
                type narrower).
        """
        offline: OfflineRuntimeConfig | None = config.offline
        if offline is None:
            msg = "OfflinePipelineContext requires an offline runtime config block"
            raise ValueError(msg)

        artifacts_root = Path(config.paths.artifacts_root).resolve()
        layout_version = feature_registry.layout_version
        return cls(
            config=config,
            seed=offline.seed,
            as_of=entropy.as_of(),
            code_version=code_version,
            config_hash=canonical_config_hash(config),
            candidate_source=candidate_source,
            embedding_model=embedding_model,
            artifact_store=artifact_store,
            entropy=entropy,
            feature_registry=feature_registry,
            artifacts_root=artifacts_root,
            data_root=Path(config.paths.data_root).resolve(),
            quarantine_root=artifacts_root / "_quarantine",
            checkpoints_root=artifacts_root / "_checkpoints",
            layout_version=layout_version,
        )