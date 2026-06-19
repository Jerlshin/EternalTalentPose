
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "VocabExpansionStage",
    "vocab_expansion_stage",
)

#: Cosine similarity at/above which a candidate term is admitted to a concept.
_EXPANSION_THRESHOLD: Final[float] = 0.55
#: Maximum terms added to any single concept (bounds semantic drift).
_MAX_EXPANSIONS_PER_CONCEPT: Final[int] = 25
#: Total vocab cap per concept after expansion (artifact-size bound).
_MAX_VOCAB_PER_CONCEPT: Final[int] = 80


class VocabExpansionStage(OfflineStage):
    """O5 — expand concept vocabularies by embedding nearest-neighbour + gate."""

    stage_id = "O5"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        dictionary = self._load_concepts(ctx)
        if not dictionary:
            msg = "vocab expansion received an empty concept dictionary from O4"
            raise ArtifactContractError(msg)

        # Build the shared candidate-term pool: the union of all vocab + seed
        # terms across concepts (deterministic, sorted). This is what we embed
        # once and reuse for every concept's nearest-neighbour search.
        candidate_terms = self._candidate_term_pool(dictionary)
        term_vectors = self._embed_terms(ctx, candidate_terms)

        expanded: dict[str, object] = {}
        total_added = 0
        for concept in sorted(dictionary):
            body = dictionary[concept]
            assert isinstance(body, Mapping)
            base_vocab = self._as_str_list(body.get("vocab"))
            centroid = self._concept_centroid(base_vocab, candidate_terms, term_vectors)
            additions = self._expand_concept(
                base_vocab, centroid, candidate_terms, term_vectors
            )
            total_added += len(additions)
            merged_vocab = self._merge_vocab(base_vocab, additions)
            expanded[concept] = {
                "vocab": merged_vocab,
                "phrases": self._as_str_list(body.get("phrases")),
                "anchor_text": self._compose_anchor_text(concept, merged_vocab, body),
                "added_terms": additions,
                "expanded": True,
            }

        artifact = self.emit_json(ctx, "concepts", {"concepts": expanded})
        metrics: dict[str, object] = {
            "concepts": len(expanded),
            "terms_added": total_added,
            "candidate_pool_size": len(candidate_terms),
            "threshold": _EXPANSION_THRESHOLD,
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    # ------------------------------------------------------------------ #
    # Input loading                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_concepts(ctx: OfflinePipelineContext) -> Mapping[str, object]:
        """Load O4's concept dictionary from the verified artifact store.

        Raises:
            ArtifactContractError: the artifact is missing the ``concepts`` key
                or is structurally malformed (the store already verified sha256).
        """
        raw = ctx.artifact_store.load_text("concepts")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"concepts.json is not valid JSON: {exc}"
            raise ArtifactContractError(msg) from exc
        if not isinstance(parsed, Mapping) or "concepts" not in parsed:
            msg = "concepts.json missing required 'concepts' mapping"
            raise ArtifactContractError(msg)
        concepts = parsed["concepts"]
        if not isinstance(concepts, Mapping):
            msg = "concepts.json 'concepts' must be a mapping"
            raise ArtifactContractError(msg)
        return concepts

    @staticmethod
    def _candidate_term_pool(dictionary: Mapping[str, object]) -> tuple[str, ...]:
        """Build the deterministic, de-duplicated union of all concept vocab terms."""
        pool: set[str] = set()
        for body in dictionary.values():
            if isinstance(body, Mapping):
                vocab = body.get("vocab")
                if isinstance(vocab, (list, tuple)):
                    pool.update(str(t) for t in vocab if isinstance(t, str) and t)
        return tuple(sorted(pool))

    # ------------------------------------------------------------------ #
    # Embedding + nearest-neighbour expansion                            #
    # ------------------------------------------------------------------ #
    def _embed_terms(
        self, ctx: OfflinePipelineContext, terms: Sequence[str]
    ) -> npt.NDArray[np.float32]:
        """Embed the candidate-term pool via the offline port; assert unit norm.

        Returns an ``(len(terms), dim)`` float32 matrix, row order == ``terms``.
        Empty pool yields an empty ``(0, dim)`` matrix (handled by callers).

        Raises:
            ArtifactContractError: the model returned a wrong-shaped or
                non-finite matrix (the embedding contract is L2-normalized rows).
        """
        if not terms:
            return np.zeros((0, ctx.embedding_model.dim), dtype=np.float32)
        vectors = ctx.embedding_model.encode(list(terms))
        if vectors.shape != (len(terms), ctx.embedding_model.dim):
            msg = (
                f"embedding model returned shape {vectors.shape}; "
                f"expected ({len(terms)}, {ctx.embedding_model.dim})"
            )
            raise ArtifactContractError(msg)
        if not np.all(np.isfinite(vectors)):
            raise ArtifactContractError("embedding model returned non-finite vectors")
        return vectors.astype(np.float32, copy=False)

    @staticmethod
    def _concept_centroid(
        base_vocab: Sequence[str],
        candidate_terms: Sequence[str],
        term_vectors: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32] | None:
        """Compute the L2-normalized centroid of a concept's existing vocab vectors.

        Returns ``None`` when the concept has no embeddable vocab (its expansion
        is then skipped — nothing to anchor the neighbourhood on).
        """
        index = {term: i for i, term in enumerate(candidate_terms)}
        rows = [index[t] for t in base_vocab if t in index]
        if not rows:
            return None
        centroid = term_vectors[rows].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0.0:
            return None
        normalized: npt.NDArray[np.float32] = (centroid / norm).astype(
            np.float32, copy=False
        )
        return normalized

    @staticmethod
    def _expand_concept(
        base_vocab: Sequence[str],
        centroid: npt.NDArray[np.float32] | None,
        candidate_terms: Sequence[str],
        term_vectors: npt.NDArray[np.float32],
    ) -> list[str]:
        """Return new terms whose cosine to the concept centroid clears the gate.

        Deterministic: scores are computed for every non-member term, filtered by
        :data:`_EXPANSION_THRESHOLD`, then sorted by ``(-cosine, term)`` and
        capped at :data:`_MAX_EXPANSIONS_PER_CONCEPT`. Cosine is a dot product
        because vectors are unit-norm.
        """
        if centroid is None or term_vectors.shape[0] == 0:
            return []
        existing = set(base_vocab)
        sims = term_vectors @ centroid  # (T,) cosine, unit-norm rows.
        scored: list[tuple[float, str]] = []
        for i, term in enumerate(candidate_terms):
            if term in existing:
                continue
            cosine = float(sims[i])
            if cosine >= _EXPANSION_THRESHOLD:
                scored.append((cosine, term))
        scored.sort(key=lambda kv: (-kv[0], kv[1]))
        return [term for _, term in scored[:_MAX_EXPANSIONS_PER_CONCEPT]]

    # ------------------------------------------------------------------ #
    # Vocab + anchor composition                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_vocab(base_vocab: Sequence[str], additions: Sequence[str]) -> list[str]:
        """Merge base + added terms, de-duplicated, sorted, capped for the artifact."""
        merged = sorted({*base_vocab, *additions})
        return merged[:_MAX_VOCAB_PER_CONCEPT]

    @staticmethod
    def _compose_anchor_text(
        concept: str, vocab: Sequence[str], body: Mapping[str, object]
    ) -> str:
        """Recompose the concept's ``anchor_text`` over its expanded vocab.

        Keeps O4's authored seed-led order where present (prefix preserved), then
        appends the expanded vocab. Never empty — falls back to the concept name.
        """
        prior = body.get("anchor_text")
        prefix = prior if isinstance(prior, str) and prior else concept
        joined = ", ".join(vocab[: _MAX_VOCAB_PER_CONCEPT])
        return f"{prefix}; {joined}" if joined else prefix

    @staticmethod
    def _as_str_list(value: object) -> list[str]:
        """Coerce a JSON value to a list of non-empty strings (defensive)."""
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if isinstance(v, str) and v]
        return []


def vocab_expansion_stage() -> VocabExpansionStage:
    """Factory: construct the O5 expansion stage bound to the frozen registry."""
    return VocabExpansionStage()