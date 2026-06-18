"""Online composition root — binds adapters to ports and runs R0-R9.

Owner layer: pipelines (online composition root).
Allowed imports: ``domain``, ``ports``, ``engines``, ``adapters`` (onnx/parquet/
csv/json/entropy/jsonl only — never ``st_embedder``), ``config``,
``observability``, this package (Repository Layout §12). Mirrors
``pipelines.offline.compose``'s charter — the CLI's ``rank`` verb hands off
here because ``cli`` may never import ``adapters`` directly (§8b).

:func:`run_online_rank` is the single entrypoint: it reads the manifest to
learn the embedding contract, constructs every online adapter (artifact
store, vector store, onnx fallback encoder, candidate source, submission +
report sinks, RNG-disabled entropy), builds the :class:`OnlineRunConfig`, and
executes the full R0→R9 lifecycle via :class:`OnlinePipeline`.

Known gap (tracked, not fixed here): ``pipelines.online.stages.r0_load`` reads
artifact keys (e.g. ``"weights/scoring_weights.locked"``,
``"calibration/logistics_weights"``, ``"archetypes/centroids"``) that do not
match the offline registry's actual manifest keys, and references a
tokenizer artifact the offline build never produces. That mismatch lives in
``stages.py``, not in this composition root, and needs its own pass once a
completed offline build exists to test against.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Final

from redstack.adapters.artifact_store_fs import FilesystemArtifactStoreAdapter
from redstack.adapters.candidate_jsonl import JsonlCandidateSourceAdapter
from redstack.adapters.entropy import OnlineEntropy
from redstack.adapters.onnx_embedder import OnnxEmbeddingModelAdapter
from redstack.adapters.run_report_json import JsonRunReportSinkAdapter
from redstack.adapters.submission_csv import CsvSubmissionSinkAdapter
from redstack.adapters.vector_store_parquet import ParquetSemanticVectorStoreAdapter
from redstack.config.determinism import build_onnx_session_options
from redstack.pipelines.context import canonical_config_hash
from redstack.pipelines.online.pipeline import (
    OnlinePipeline,
    OnlinePipelineResult,
    OnlineRunConfig,
)
from redstack.ports._types import ArtifactKey

if TYPE_CHECKING:
    from redstack.config.schema import RedstackConfig

__all__: tuple[str, ...] = ("run_online_rank",)

#: Placeholder tokenizer key — no offline stage produces this artifact today
#: (see module docstring "Known gap"); kept as a single named constant so the
#: eventual fix has one call site to update.
_TOKENIZER_KEY: Final[ArtifactKey] = ArtifactKey("tokenizer")


def _sha256_file(path: Path) -> str:
    """Stream ``path``'s sha256 without loading it fully into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_online_rank(
    config: RedstackConfig,
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    participant_id: str,
) -> OnlinePipelineResult:
    """Bind adapters, build ``OnlineRunConfig``, and execute R0→R9.

    Args:
        config: The fully-composed, validated online ``RedstackConfig``.
        input_path: The candidates ``.jsonl`` to rank.
        output_path: Destination ``submission.csv`` path.
        report_path: Destination ``run_report.json`` path.
        participant_id: Output filename stem recorded into the run config.

    Returns:
        The terminal :class:`OnlinePipelineResult`.

    Raises:
        ValueError: ``config.online`` is absent (wrong run mode).
    """
    online = config.online
    if online is None:
        msg = "run_online_rank requires a config with an 'online' runtime block"
        raise ValueError(msg)

    artifacts_root = Path(config.paths.artifacts_root).resolve()
    artifact_store = FilesystemArtifactStoreAdapter(artifacts_root)
    manifest = artifact_store.manifest()

    run_config = OnlineRunConfig(
        participant_id=participant_id,
        code_version=_code_version(),
        config_hash=canonical_config_hash(config),
        seed=config.determinism.seed,
        floor_sentinel=online.score_presentation.floor_sentinel,
        ranking_size=online.top_k,
        score_decimals=online.score_presentation.decimals,
        abort_on_malformed=online.malformed_record_policy.value == "abort",
        budget_limit_seconds=config.budget.max_wall_seconds,
        blas_threads=config.determinism.omp_num_threads,
        onnx_intra_op_threads=config.determinism.onnx_intra_op_threads,
        onnx_inter_op_threads=config.determinism.onnx_inter_op_threads,
    )

    candidate_source = JsonlCandidateSourceAdapter(input_path)
    vector_store = ParquetSemanticVectorStoreAdapter(
        artifact_store,
        vectors_key=ArtifactKey("candidate_vectors"),
        expected_dim=manifest.embedding.dim,
    )
    session_options = build_onnx_session_options(config.determinism)
    embedding_model = OnnxEmbeddingModelAdapter(
        artifact_store,
        model_key=ArtifactKey("encoder"),
        tokenizer_key=_TOKENIZER_KEY,
        model_id=manifest.embedding.model_id,
        dim=manifest.embedding.dim,
        session_options=session_options,
    )
    submission_sink = CsvSubmissionSinkAdapter(
        output_path, score_decimals=online.score_presentation.decimals
    )
    run_report_sink = JsonRunReportSinkAdapter(report_path)
    entropy = OnlineEntropy(seed=config.determinism.seed, as_of=online.as_of.date())

    pipeline = OnlinePipeline(
        config=run_config,
        candidate_source=candidate_source,
        vector_store=vector_store,
        embedding_model=embedding_model,
        submission_sink=submission_sink,
        run_report_sink=run_report_sink,
        entropy=entropy,
        artifact_store=artifact_store,
    )
    return pipeline.run(input_file_sha256=_sha256_file(input_path))


def _code_version() -> str:
    """Return the run's code provenance for the report (package version)."""
    import redstack

    return redstack.__version__
