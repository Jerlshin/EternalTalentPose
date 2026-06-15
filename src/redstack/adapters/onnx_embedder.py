"""``OnnxEmbeddingModelAdapter`` — implements ``EmbeddingModelPort`` (Adapters §3).

Owner layer: adapters (infrastructure — impure compute).
Allowed imports: stdlib typing/importlib; ``onnxruntime``; numpy; ``domain.errors``;
``ports``.
Forbidden: ``engines``, ``pipelines``, the network, any model download.

The online *fallback* encoder: turns composed documents into normalized dense
vectors via ONNX Runtime, CPU-only, deterministic, no network. Invoked only for
``CandidateId``s missing from the vector store. The onnx graph is the exported
twin of the offline base model (same ``model_id``/``dim``); it emits token
embeddings (``last_hidden_state``) and this adapter applies the fixed pooling and
L2 normalization, or accepts an already-pooled 2-D output and only normalizes.

Determinism: the injected ``SessionOptions`` pin intra-/inter-op threads and
sequential execution (built once by ``config.determinism`` in the composition
root); the only execution provider registered is ``CPUExecutionProvider``.
Pooling/normalization are fixed, float32, no sampling — bitwise-deterministic
within this runtime; cross-runtime vs the offline sentence-transformers vectors
is cosine-within-ε by contract (Ports §9). The model and tokenizer are local,
hash-verified artifacts reached through the injected ``ArtifactStorePort`` — no
network is ever touched.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt

from redstack.domain.errors import ArtifactContractError
from redstack.ports._types import ArtifactKey, FloatMatrix
from redstack.ports.artifact_store import ArtifactStorePort
from redstack.ports.embedding import EmbeddingError

#: Integer token-feed matrix, shape ``(batch, seq)``.
type _I64Matrix = npt.NDArray[np.int64]

#: The only execution provider permitted online (CPU-only, network-isolated).
_CPU_PROVIDER: Final[str] = "CPUExecutionProvider"
#: Default per-call batch size (throughput hint; never affects results).
_DEFAULT_BATCH_SIZE: Final[int] = 32
#: Token-embedding input names the adapter knows how to populate.
_IDS_NAMES: Final[frozenset[str]] = frozenset({"input_ids"})
_MASK_NAMES: Final[frozenset[str]] = frozenset({"attention_mask"})
_TYPE_NAMES: Final[frozenset[str]] = frozenset({"token_type_ids", "segment_ids"})

PoolingMode = Literal["mean", "cls"]


# --------------------------------------------------------------------------- #
# Minimal structural views over the untyped ``tokenizers`` library (loaded via
# importlib so no untyped ``import`` statement enters the typed surface).
# --------------------------------------------------------------------------- #
class _Encoding(Protocol):
    @property
    def ids(self) -> Sequence[int]: ...
    @property
    def attention_mask(self) -> Sequence[int]: ...


class _Tokenizer(Protocol):
    def enable_truncation(self, max_length: int) -> None: ...
    def enable_padding(self) -> None: ...
    def encode_batch(self, input: Sequence[str]) -> list[_Encoding]: ...


def _load_tokenizer(json_text: str) -> _Tokenizer:
    """Build a fast tokenizer from hash-verified ``tokenizer.json`` text.

    Raises:
        EmbeddingError: the ``tokenizers`` runtime is unavailable or the
            tokenizer payload cannot be parsed.
    """
    try:
        module = importlib.import_module("tokenizers")
        tokenizer_cls = module.Tokenizer
        instance = tokenizer_cls.from_str(json_text)
    except (ImportError, ValueError, TypeError) as exc:
        raise EmbeddingError(f"cannot construct tokenizer: {exc}") from exc
    return cast("_Tokenizer", instance)


class _OrtNodeArg(Protocol):
    @property
    def name(self) -> str: ...


class _OrtSession(Protocol):
    def get_inputs(self) -> list[_OrtNodeArg]: ...
    def get_outputs(self) -> list[_OrtNodeArg]: ...
    def run(
        self, output_names: list[str], input_feed: Mapping[str, _I64Matrix]
    ) -> list[npt.NDArray[np.float32]]: ...


def _create_session(model_path: Path, session_options: object, key: ArtifactKey) -> _OrtSession:
    """Create a CPU-only, thread-pinned ONNX Runtime session (no stub dependency).

    ``onnxruntime`` ships no ``py.typed``; it is loaded via ``importlib`` and the
    session narrowed to :class:`_OrtSession` so no ``Any`` enters the adapter.

    Raises:
        EmbeddingError: the session could not be created.
    """
    try:
        ort_module = importlib.import_module("onnxruntime")
        session = ort_module.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[_CPU_PROVIDER],
        )
    except Exception as exc:  # onnxruntime raises bare exceptions on load
        raise EmbeddingError(
            f"cannot create onnx session for {key!r}: {exc}"
        ) from exc
    return cast("_OrtSession", session)


class OnnxEmbeddingModelAdapter:
    """CPU-only, thread-pinned ONNX Runtime fallback encoder.

    Constructed at R0 from a hash-verified locator obtained via the injected
    ``ArtifactStorePort``; creates one ``InferenceSession`` with pinned options
    and serves ``encode`` for the run. Not shared mutably across threads.
    """

    __slots__ = (
        "_model_id",
        "_dim",
        "_pooling",
        "_max_seq_length",
        "_session",
        "_tokenizer",
        "_input_names",
        "_output_name",
    )

    def __init__(
        self,
        store: ArtifactStorePort,
        *,
        model_key: ArtifactKey,
        tokenizer_key: ArtifactKey,
        model_id: str,
        dim: int,
        session_options: object,
        max_seq_length: int = 256,
        pooling: PoolingMode = "mean",
    ) -> None:
        """Open the onnx session + tokenizer and assert dimensionality.

        Args:
            store: Injected artifact store; ``locate(model_key)`` yields the
                verified onnx path and ``load_text(tokenizer_key)`` the verified
                tokenizer payload.
            model_key: Manifest key of ``model/encoder.onnx``.
            tokenizer_key: Manifest key of the hash-pinned tokenizer asset.
            model_id: The contracted model identity (from the manifest).
            dim: The contracted embedding dimensionality (from the manifest).
            session_options: Opaque, pinned, sequential CPU session-options
                handle built by ``config.determinism`` in the composition root
                and forwarded untouched to the ONNX Runtime session.
            max_seq_length: Truncation length for tokenization.
            pooling: Token-embedding pooling for 3-D outputs.

        Raises:
            EmbeddingError: the onnx session or tokenizer cannot be created.
            ArtifactContractError: the produced width disagrees with ``dim``.
        """
        self._model_id: Final[str] = model_id
        self._dim: Final[int] = dim
        self._pooling: Final[PoolingMode] = pooling
        self._max_seq_length: Final[int] = max_seq_length

        locator = store.locate(model_key)
        model_path = self._locator_path(locator.opaque_handle, model_key)
        self._session: Final[_OrtSession] = _create_session(
            model_path, session_options, model_key
        )

        tokenizer = _load_tokenizer(store.load_text(tokenizer_key))
        tokenizer.enable_truncation(max_seq_length)
        tokenizer.enable_padding()
        self._tokenizer: Final[_Tokenizer] = tokenizer

        self._input_names: Final[tuple[str, ...]] = tuple(
            node.name for node in self._session.get_inputs()
        )
        outputs = self._session.get_outputs()
        if not outputs:
            raise EmbeddingError(f"onnx model {model_key!r} declares no outputs")
        self._output_name: Final[str] = outputs[0].name

        probe = self._run_batch(["probe"])
        if probe.shape[1] != self._dim:
            raise ArtifactContractError(
                f"onnx output width {probe.shape[1]} disagrees with manifest "
                f"dim {self._dim} for {model_key!r}"
            )

    # ------------------------------------------------------------------ #
    # Construction helpers.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _locator_path(handle: object, key: ArtifactKey) -> Path:
        if isinstance(handle, Path):
            return handle
        if isinstance(handle, str):
            return Path(handle)
        raise EmbeddingError(
            f"locator for {key!r} carries a non-path handle: {type(handle).__name__}"
        )

    # ------------------------------------------------------------------ #
    # Port surface.
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        """The fixed output dimensionality."""
        return self._dim

    @property
    def model_id(self) -> str:
        """The stable model identifier for provenance."""
        return self._model_id

    def encode(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> FloatMatrix:
        """Encode pre-composed documents into a read-only ``(len(texts), dim)`` matrix.

        Output is ``float32``, each row L2-normalized within epsilon, row order
        equal to input order regardless of batching.

        Raises:
            EmbeddingError: tokenization or onnx execution failed.
        """
        n = len(texts)
        if n == 0:
            empty = np.empty((0, self._dim), dtype=np.float32)
            empty.flags.writeable = False
            return empty

        step = batch_size if batch_size is not None and batch_size > 0 else _DEFAULT_BATCH_SIZE
        blocks: list[FloatMatrix] = []
        for start in range(0, n, step):
            blocks.append(self._run_batch(list(texts[start : start + step])))

        matrix: FloatMatrix = (
            blocks[0] if len(blocks) == 1 else np.vstack(blocks).astype(np.float32, copy=False)
        )
        if matrix.shape != (n, self._dim):
            raise EmbeddingError(
                f"encoded shape {matrix.shape} != expected {(n, self._dim)}"
            )
        matrix.flags.writeable = False
        return matrix

    # ------------------------------------------------------------------ #
    # Core encode (tokenize -> session -> pool -> normalize).
    # ------------------------------------------------------------------ #
    def _run_batch(self, batch: list[str]) -> FloatMatrix:
        try:
            encodings = self._tokenizer.encode_batch(batch)
        except Exception as exc:  # tokenizers raises bare exceptions
            raise EmbeddingError(f"tokenization failed: {exc}") from exc

        input_ids = np.asarray([list(enc.ids) for enc in encodings], dtype=np.int64)
        attention_mask = np.asarray(
            [list(enc.attention_mask) for enc in encodings], dtype=np.int64
        )
        feed = self._build_feed(input_ids, attention_mask)

        try:
            raw_outputs = self._session.run([self._output_name], feed)
        except Exception as exc:  # onnxruntime raises bare exceptions on run
            raise EmbeddingError(f"onnx inference failed: {exc}") from exc

        token_output = np.asarray(raw_outputs[0], dtype=np.float32)
        pooled = self._pool(token_output, attention_mask)
        return self._l2_normalize(pooled)

    def _build_feed(
        self, input_ids: _I64Matrix, attention_mask: _I64Matrix
    ) -> dict[str, _I64Matrix]:
        feed: dict[str, _I64Matrix] = {}
        for name in self._input_names:
            if name in _IDS_NAMES:
                feed[name] = input_ids
            elif name in _MASK_NAMES:
                feed[name] = attention_mask
            elif name in _TYPE_NAMES:
                feed[name] = np.zeros_like(input_ids)
            else:
                raise EmbeddingError(
                    f"onnx model requires unsupported input {name!r}"
                )
        return feed

    def _pool(self, token_output: FloatMatrix, attention_mask: _I64Matrix) -> FloatMatrix:
        if token_output.ndim == 2:
            # Already sentence-level (the export pooled internally).
            return token_output.astype(np.float32, copy=False)
        if token_output.ndim != 3:
            raise EmbeddingError(
                f"unexpected onnx output rank {token_output.ndim}; expected 2 or 3"
            )
        if self._pooling == "cls":
            return token_output[:, 0, :].astype(np.float32, copy=False)
        mask = attention_mask.astype(np.float32)[:, :, None]
        summed = np.sum(token_output * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled: FloatMatrix = (summed / counts).astype(np.float32, copy=False)
        return pooled

    @staticmethod
    def _l2_normalize(matrix: FloatMatrix) -> FloatMatrix:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe = np.clip(norms, a_min=1e-12, a_max=None)
        normalized: FloatMatrix = (matrix / safe).astype(np.float32, copy=False)
        return normalized


if TYPE_CHECKING:
    from redstack.ports.embedding import EmbeddingModelPort

    # Compile-time structural conformance to the frozen port surface.
    _PORT_CONFORMANCE: type[EmbeddingModelPort] = OnnxEmbeddingModelAdapter


__all__: tuple[str, ...] = ("OnnxEmbeddingModelAdapter", "PoolingMode")