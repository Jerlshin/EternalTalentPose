"""``OfflinePipelineContext`` — the resolved environment for one offline build.

Owner layer: pipelines (offline composition root).
Allowed imports: ``domain``, ``ports``, ``config.schema``, ``features``,
stdlib (Repository Layout §12). Adapters are *instantiated* by the offline
``pipeline.py`` composition root and arrive here already bound to their ports;
this carrier never constructs them.

Per the Offline Pipeline spec (Part 1, ``OfflinePipelineContext``): the context
is built once by the offline composition root and is immutable thereafter. It
carries the config, the bound ports (``CandidateSource``, ``EmbeddingModel`` —
the sentence-transformers backend offline, ``ArtifactStore``, ``OfflineEntropy``),
``seed``, ``as_of``, output roots, and the ``FeatureRegistry``/``FeatureLayout``
constant. It records ``config_hash``, ``seed``, ``as_of``, ``code_version`` for
the report. It owns no IO — it is purely a carrier handed to stage callables.

The roots model the hard directory line (Repository Layout §0): ``data_root``
and a read-only ``configs`` view are *inputs*; ``artifacts_root`` is the single
write target; ``quarantine_root`` and ``checkpoints_root`` are runner-internal
scratch under the artifacts tree, never manifested.
"""

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
    """Immutable resolved build environment for O0–O18.

    Extends the shared :class:`RunContext` (config + reproducibility provenance)
    with the bound ports, output roots, and the layout constant a build needs.
    Construct via :meth:`build` from the offline composition root; never mutate.

    Attributes:
        candidate_source: Streams raw candidate records (O0/O1/O2/O5/O13a).
        embedding_model: The offline ``sentence-transformers`` backend behind
            ``EmbeddingModelPort`` (O13 candidate/anchor vectors). Import-guarded
            to the offline path.
        artifact_store: Verifies the freshly built artifact set at O17/O18.
        entropy: The seeded ``OfflineEntropy`` — KMeans init (O7), weight search
            (O9), sampling/splits (O8). Distinct labels yield independent streams.
        feature_registry: The populated ordered feature registry carrying the
            shared ``layout_version`` (Repository Layout §6/§8). The CQV/feature
            snapshot (O14) and weight calibration (O9) bind to it.
        artifacts_root: The single write target for produced artifacts.
        data_root: Read-only raw input root.
        quarantine_root: Where the runner moves partial/failed stage output so it
            is never manifested (Offline Pipeline Part 11).
        checkpoints_root: Where the runner persists ``StageReceipt``s + outputs
            for resume.
        layout_version: The shared layout version mirrored from the registry,
            pinned for the manifest's ``layout_version`` coherence check.
    """

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