"""O13 — Embedding Generation (O13a candidates, O13b anchors, O13c careers, O13 manifest+onnx).

Owner stages: O13a / O13b / O13c / O13 (Offline Pipeline Part 9; Adapters §4).
Produce all dense vectors + the ONNX twin + the embedding manifest — the
compute-dominant offline stage. The text→vector work goes through the offline
``EmbeddingModelPort`` (the sentence-transformers adapter); the ONNX export +
st↔onnx parity is the adapter's responsibility (Adapters §4), reached via the
optional ``OnnxExportCapable`` capability so the frozen port stays unchanged.

Sub-stage map (graph nodes are distinct so the runner schedules/caches each):

* **O13a** (deps O1) — stream the pool, compose each candidate's embedding
  document via the pinned O1 recipe, batch-encode through the port, write
  ``candidate_vectors.parquet`` (id→row, unit-norm, unique ids, mmap-ready).
* **O13b** (deps O6) — encode the O6 positive+negative anchor texts, write
  ``anchor_vectors.npy`` ((a, dim), row order == sorted anchor id; keys ⊆
  jd_concepts so §8 coherence holds).
* **O13c** (deps O1, optional/non-critical) — per-role career-history vectors,
  ``career_vectors.parquet`` (finer R3 matching). Non-critical: a failure is
  flagged, the build continues (graph marks O13c non-critical).
* **O13** (deps O13a/b/c) — export ``model/encoder.onnx`` (parity-verified) and
  write ``embedding_manifest.json`` (model_id, dim, **recipe** byte-matching O1,
  pooling) — the manifest the online fallback and §8 coherence bind to.

Determinism (Part 9): within-runtime bitwise; the doc-composition recipe is
pinned (``normalization.EMBEDDING_DOC_RECIPE``) so the online onnx fallback lands
in the same space; vectors are float32, contiguous, L2-normalized; row/key order
is deterministic. Encoding streams the pool with bounded batch buffers (O(batch)
memory, never the whole 100K decoded set held at once).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage
from redstack.pipelines.offline.stages.normalization import (
    EMBEDDING_DOC_RECIPE_VERSION,
    compose_embedding_document,
)
from redstack.ports._types import SourceMalformed, SourceOk
from redstack.ports.embedding import OnnxExportCapable

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "CandidateEmbeddingStage",
    "AnchorEmbeddingStage",
    "CareerEmbeddingStage",
    "EmbeddingManifestStage",
    "candidate_embedding_stage",
    "anchor_embedding_stage",
    "career_embedding_stage",
    "embedding_manifest_stage",
)

#: Encode batch size handed to the port (an adapter hint; never changes results).
_ENCODE_BATCH: Final[int] = 256
#: Sub-stage flush size: encode + write in chunks to bound peak memory on 100K.
_FLUSH_EVERY: Final[int] = 8192
#: Default pooling label recorded in the manifest (model-fixed; informational).
_POOLING: Final[str] = "mean"
#: Sample size used for the onnx parity check (Adapters §4).
_PARITY_SAMPLE: Final[int] = 64


def _l2_normalize(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return ``matrix`` with each row L2-normalized (float32, contiguous).

    Zero rows (degenerate empty-doc encodes) are left as zeros rather than
    divided — the embedding adapter never returns zeros to mask failure, so a
    true zero row is a defensive edge, not the normal path. Callers that require
    strict unit norm (the parquet/npy emit helpers) re-assert it.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    normalized = (matrix / safe).astype(np.float32, copy=False)
    return np.ascontiguousarray(normalized)


class _EmbeddingStageBase(OfflineStage):
    """Shared batched-encode helpers for the O13 family."""

    def _encode_batch(
        self, ctx: OfflinePipelineContext, texts: Sequence[str]
    ) -> npt.NDArray[np.float32]:
        """Encode a batch through the port and validate the embedding contract.

        Raises:
            ArtifactContractError: the model returns a wrong-shaped / non-float32
                / non-finite matrix (the encode contract is L2-normalized rows;
                we re-normalize defensively to guarantee the artifact invariant).
        """
        if not texts:
            return np.zeros((0, ctx.embedding_model.dim), dtype=np.float32)
        vectors = ctx.embedding_model.encode(list(texts), batch_size=_ENCODE_BATCH)
        expected = (len(texts), ctx.embedding_model.dim)
        if vectors.shape != expected:
            msg = f"embedding model returned shape {vectors.shape}; expected {expected}"
            raise ArtifactContractError(msg)
        if not np.all(np.isfinite(vectors)):
            raise ArtifactContractError("embedding model returned non-finite vectors")
        return _l2_normalize(vectors.astype(np.float32, copy=False))


class CandidateEmbeddingStage(_EmbeddingStageBase):
    """O13a — stream the pool, encode composed docs → ``candidate_vectors.parquet``."""

    stage_id = "O13a"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        ids: list[str] = []
        chunks: list[npt.NDArray[np.float32]] = []
        pending_ids: list[str] = []
        pending_docs: list[str] = []
        encoded = 0

        def flush() -> None:
            nonlocal encoded
            if not pending_docs:
                return
            block = self._encode_batch(ctx, pending_docs)
            chunks.append(block)
            ids.extend(pending_ids)
            encoded += len(pending_docs)
            pending_ids.clear()
            pending_docs.clear()

        for record in ctx.candidate_source.stream():
            if isinstance(record, SourceMalformed):
                continue
            if not isinstance(record, SourceOk):
                continue
            raw = record.raw
            cid = raw.get("candidate_id")
            if not isinstance(cid, str) or not cid:
                continue
            pending_ids.append(cid)
            pending_docs.append(compose_embedding_document(raw))
            if len(pending_docs) >= _FLUSH_EVERY:
                flush()
        flush()

        if not ids:
            msg = "candidate embedding generation saw zero candidates"
            raise ArtifactContractError(msg)

        vectors = (
            np.ascontiguousarray(np.concatenate(chunks, axis=0))
            if len(chunks) > 1
            else chunks[0]
        )
        artifact = self.emit_vector_parquet(
            ctx, "candidate_vectors", ids=ids, vectors=vectors
        )
        metrics: dict[str, object] = {
            "encoded_candidates": encoded,
            "dim": int(vectors.shape[1]),
            "model_id": ctx.embedding_model.model_id,
            "recipe_version": EMBEDDING_DOC_RECIPE_VERSION,
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)


class AnchorEmbeddingStage(_EmbeddingStageBase):
    """O13b — encode O6 anchor texts → ``anchor_vectors.npy`` (keys ⊆ jd_concepts)."""

    stage_id = "O13b"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        anchors = self._load_anchor_texts(ctx)
        if not anchors:
            msg = "anchor embedding generation found no anchors in jd_concepts"
            raise ArtifactContractError(msg)
        # Deterministic key order: sorted anchor id → row index.
        anchor_ids = sorted(anchors)
        texts = [anchors[aid] for aid in anchor_ids]
        vectors = self._encode_batch(ctx, texts)
        vectors = np.ascontiguousarray(vectors)

        artifact = self.emit_array(ctx, "anchor_vectors", vectors)
        metrics: dict[str, object] = {
            "anchor_count": len(anchor_ids),
            "dim": int(vectors.shape[1]) if vectors.shape[0] else ctx.embedding_model.dim,
            "anchor_ids": anchor_ids,
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    @staticmethod
    def _load_anchor_texts(ctx: OfflinePipelineContext) -> dict[str, str]:
        """Load O6's ``jd_concepts.json`` and return ``{anchor_id: text}``.

        Raises:
            ArtifactContractError: jd_concepts is missing/malformed or an anchor
                lacks an id/text (the anchor set defines the npy key universe; a
                broken anchor would silently shrink R3 coverage).
        """
        try:
            raw = ctx.artifact_store.load_text("jd_concepts")
        except Exception as exc:  # noqa: BLE001 — surface as a contract failure.
            msg = f"cannot load jd_concepts for anchor embedding: {exc}"
            raise ArtifactContractError(msg) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"jd_concepts.json is not valid JSON: {exc}"
            raise ArtifactContractError(msg) from exc
        if not isinstance(parsed, Mapping):
            raise ArtifactContractError("jd_concepts.json must be an object")
        anchors = parsed.get("anchors")
        if not isinstance(anchors, (list, tuple)) or not anchors:
            raise ArtifactContractError("jd_concepts.json has no anchors")
        texts: dict[str, str] = {}
        for anchor in anchors:
            if not isinstance(anchor, Mapping):
                raise ArtifactContractError("each anchor must be an object")
            aid = anchor.get("id")
            text = anchor.get("text")
            if not isinstance(aid, str) or not aid:
                raise ArtifactContractError("anchor missing a string id")
            if not isinstance(text, str) or not text:
                raise ArtifactContractError(f"anchor {aid!r} missing text")
            texts[aid] = text
        return texts


class CareerEmbeddingStage(_EmbeddingStageBase):
    """O13c — optional per-role career vectors → ``career_vectors.parquet``.

    Non-critical (graph-flagged): finer R3 detail matching. This sub-stage is not
    in the frozen artifact registry's required-online set, so when a
    ``career_vectors`` key is absent from the registry it emits no artifact and
    records its intent in metrics — keeping the build green while the optional
    artifact path is unconfigured. When the key exists it writes a vector parquet
    of ``"{candidate_id}#{role_index}"`` rows.
    """

    stage_id = "O13c"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        if "career_vectors" not in self.registry.keys():
            # Optional artifact not configured in this registry build — emit
            # nothing (O13c is non-critical) but record the decision.
            return StageResult(
                artifacts=(),
                metrics={"career_vectors": "not_configured", "encoded_roles": 0},
            )

        ids: list[str] = []
        chunks: list[npt.NDArray[np.float32]] = []
        pending_ids: list[str] = []
        pending_docs: list[str] = []
        encoded = 0

        def flush() -> None:
            nonlocal encoded
            if not pending_docs:
                return
            chunks.append(self._encode_batch(ctx, pending_docs))
            ids.extend(pending_ids)
            encoded += len(pending_docs)
            pending_ids.clear()
            pending_docs.clear()

        for record in ctx.candidate_source.stream():
            if isinstance(record, SourceMalformed) or not isinstance(record, SourceOk):
                continue
            raw = record.raw
            cid = raw.get("candidate_id")
            history = raw.get("career_history")
            if not isinstance(cid, str) or not isinstance(history, (list, tuple)):
                continue
            for role_index, role in enumerate(history):
                if not isinstance(role, Mapping):
                    continue
                doc = self._compose_role_doc(role)
                if not doc:
                    continue
                pending_ids.append(f"{cid}#{role_index}")
                pending_docs.append(doc)
                if len(pending_docs) >= _FLUSH_EVERY:
                    flush()
        flush()

        if not ids:
            return StageResult(
                artifacts=(),
                metrics={"career_vectors": "no_roles", "encoded_roles": 0},
            )
        vectors = (
            np.ascontiguousarray(np.concatenate(chunks, axis=0))
            if len(chunks) > 1
            else chunks[0]
        )
        artifact = self.emit_vector_parquet(
            ctx, "career_vectors", ids=ids, vectors=vectors
        )
        return StageResult(
            artifacts=(artifact,),
            metrics={"encoded_roles": encoded, "dim": int(vectors.shape[1])},
        )

    @staticmethod
    def _compose_role_doc(role: Mapping[str, object]) -> str:
        """Compose a deterministic per-role document (title + description)."""
        parts: list[str] = []
        for key in ("title", "description"):
            value = role.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        return " ".join(parts)


class EmbeddingManifestStage(_EmbeddingStageBase):
    """O13 — export the ONNX twin (parity-verified) + write ``embedding_manifest.json``."""

    stage_id = "O13"
    #: 1.1: also emits the "tokenizer" artifact the online onnx fallback needs.
    stage_version = "1.1"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        model = ctx.embedding_model
        dim = model.dim
        artifacts = []

        # Export + register the ONNX twin when the offline adapter supports it.
        parity_cosine: float | None = None
        opset: int | None = None
        if isinstance(model, OnnxExportCapable):
            onnx_tmp = ctx.artifacts_root / "model" / "_encoder.export.onnx"
            onnx_tmp.parent.mkdir(parents=True, exist_ok=True)
            sample = self._parity_sample(ctx)
            parity_cosine = model.export_onnx(onnx_tmp, sample_texts=sample)
            opset = model.opset
            artifacts.append(
                self.emit_encoder(
                    ctx,
                    "encoder",
                    onnx_path=onnx_tmp,
                    model_id=model.model_id,
                    dim=dim,
                    parity_cosine=parity_cosine,
                )
            )
            onnx_tmp.unlink(missing_ok=True)
            # The onnx fallback encoder tokenizes through this exact fast
            # tokenizer; it must travel as its own artifact (Adapters §4).
            tokenizer_payload = json.loads(model.tokenizer_json)
            artifacts.append(self.emit_json(ctx, "tokenizer", tokenizer_payload))
        else:
            msg = (
                "offline embedding model does not support ONNX export "
                "(OnnxExportCapable); encoder.onnx cannot be produced"
            )
            raise ArtifactContractError(msg)

        manifest: dict[str, object] = {
            "model_id": model.model_id,
            "dim": dim,
            "recipe": EMBEDDING_DOC_RECIPE_VERSION,
            "pooling": _POOLING,
            "normalization": "l2",
            "opset": opset,
            "parity_cosine": round(parity_cosine, 6),
        }
        artifacts.append(self.emit_json(ctx, "embedding_manifest", manifest))
        metrics: dict[str, object] = {
            "model_id": model.model_id,
            "dim": dim,
            "opset": opset,
            "parity_cosine": round(parity_cosine, 6),
            "recipe": EMBEDDING_DOC_RECIPE_VERSION,
        }
        return StageResult(artifacts=tuple(artifacts), metrics=metrics)

    @staticmethod
    def _parity_sample(ctx: OfflinePipelineContext) -> list[str]:
        """Draw a small deterministic sample of composed docs for parity checking.

        Takes the first :data:`_PARITY_SAMPLE` composed candidate documents in
        stream order (deterministic). The adapter compares its st vs onnx encode
        of these to compute the parity cosine (Adapters §4 / Ports §9).
        """
        sample: list[str] = []
        for record in ctx.candidate_source.stream():
            if isinstance(record, SourceMalformed) or not isinstance(record, SourceOk):
                continue
            doc = compose_embedding_document(record.raw)
            if doc:
                sample.append(doc)
            if len(sample) >= _PARITY_SAMPLE:
                break
        if not sample:
            # Defensive: still allow export with a trivial probe if the pool is
            # empty in a degenerate test; the adapter handles empty samples.
            sample.append("")
        return sample


def candidate_embedding_stage() -> CandidateEmbeddingStage:
    """Factory: construct the O13a candidate-vector stage."""
    return CandidateEmbeddingStage()


def anchor_embedding_stage() -> AnchorEmbeddingStage:
    """Factory: construct the O13b anchor-vector stage."""
    return AnchorEmbeddingStage()


def career_embedding_stage() -> CareerEmbeddingStage:
    """Factory: construct the O13c (optional) career-vector stage."""
    return CareerEmbeddingStage()


def embedding_manifest_stage() -> EmbeddingManifestStage:
    """Factory: construct the O13 manifest + onnx-export stage."""
    return EmbeddingManifestStage()