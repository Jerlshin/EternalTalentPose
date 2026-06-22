
from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import numpy as np
import numpy.typing as npt

from redstack.ports._types import FloatMatrix
from redstack.ports.embedding import EmbeddingError

#: Environment variables whose value ``"online"`` forbids importing this module.
_ONLINE_PROFILE_ENVS: Final[tuple[str, ...]] = (
    "REDSTACK_EXECUTION_PROFILE",
    "REDSTACK_RUNTIME_PROFILE",
)
#: A boolean online flag whose truthy value also forbids import.
_ONLINE_FLAG_ENV: Final[str] = "REDSTACK_ONLINE"
_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
#: Default per-call batch size (throughput hint; never affects results).
_DEFAULT_BATCH_SIZE: Final[int] = 32
#: Default ONNX export opset and the accepted parity floor.
_DEFAULT_OPSET: Final[int] = 17
_PARITY_FLOOR: Final[float] = 0.999

#: Explicit override for auto device selection (build-time only; never read by
#: anything under pipelines.online). Unset means "auto-detect".
_DEVICE_ENV_VAR: Final[str] = "REDSTACK_OFFLINE_DEVICE"
_VALID_DEVICES: Final[frozenset[str]] = frozenset({"cpu", "mps", "cuda"})


def _select_device(torch_module: _Torch) -> str:
    """Resolve the offline encode device: override, else best available accelerator.

    Priority: an explicit ``REDSTACK_OFFLINE_DEVICE`` env override, then CUDA, then
    Apple MPS, else CPU. This picks the device for ``encode`` only —
    ``export_onnx`` always traces on CPU regardless of this choice (Adapters §4).

    Raises:
        EmbeddingError: the env override names a device outside ``_VALID_DEVICES``.
    """
    override = os.environ.get(_DEVICE_ENV_VAR, "").strip().lower()
    if override:
        if override not in _VALID_DEVICES:
            raise EmbeddingError(
                f"{_DEVICE_ENV_VAR}={override!r} must be one of "
                f"{sorted(_VALID_DEVICES)}"
            )
        return override
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _guard_offline_only() -> None:
    """Raise if an online execution marker is set (import + construction guard).

    Raises:
        RuntimeError: an online profile/flag is present in the environment.
    """
    for key in _ONLINE_PROFILE_ENVS:
        if os.environ.get(key, "").strip().lower() == "online":
            raise RuntimeError(
                f"adapters.st_embedder is offline-only but {key}=online is set"
            )
    if os.environ.get(_ONLINE_FLAG_ENV, "").strip().lower() in _TRUTHY:
        raise RuntimeError(
            f"adapters.st_embedder is offline-only but {_ONLINE_FLAG_ENV} is truthy"
        )


# Import-time guard (defence in depth alongside the import-linter contract).
_guard_offline_only()


# --------------------------------------------------------------------------- #
# Minimal structural views over the untyped runtimes (loaded via importlib so no
# untyped ``import torch`` / ``import sentence_transformers`` statement enters
# the typed surface; concrete objects are narrowed by ``cast``).
# --------------------------------------------------------------------------- #
class _StModel(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> FloatMatrix: ...
    def get_sentence_embedding_dimension(self) -> int: ...
    def __getitem__(self, index: int) -> object: ...
    @property
    def tokenizer(self) -> _HfTokenizer: ...


class _FastTokenizerHandle(Protocol):
    def to_str(self) -> str: ...


class _HfTokenizer(Protocol):
    def __call__(
        self,
        text: Sequence[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, object]: ...
    @property
    def backend_tokenizer(self) -> _FastTokenizerHandle: ...


class _TorchModule(Protocol):
    def to(self, device: str) -> _TorchModule: ...


class _Pooling(Protocol):
    @property
    def auto_model(self) -> _TorchModule: ...


class _TorchOnnx(Protocol):
    def export(
        self,
        model: object,
        args: object,
        f: str,
        *,
        input_names: list[str],
        output_names: list[str],
        dynamic_axes: Mapping[str, Mapping[int, str]],
        opset_version: int,
        do_constant_folding: bool,
        dynamo: bool,
    ) -> None: ...


class _TorchAccelerator(Protocol):
    def is_available(self) -> bool: ...


class _TorchBackends(Protocol):
    @property
    def mps(self) -> _TorchAccelerator: ...


class _Torch(Protocol):
    def set_num_threads(self, n: int) -> None: ...
    @property
    def onnx(self) -> _TorchOnnx: ...
    @property
    def cuda(self) -> _TorchAccelerator: ...
    @property
    def backends(self) -> _TorchBackends: ...


class _OrtSession(Protocol):
    def run(
        self, output_names: list[str], input_feed: Mapping[str, npt.NDArray[np.int64]]
    ) -> list[npt.NDArray[np.float32]]: ...


class SentenceTransformerEmbeddingAdapter:
    """Offline sentence-transformers encoder + ONNX-twin exporter.

    Constructed only inside ``pipelines/offline``. Loads the pinned-revision
    model in eval mode under the HuggingFace offline environment; serves
    ``encode`` and exports the onnx twin with parity verification.
    """

    __slots__ = (
        "_batch_size",
        "_device",
        "_dim",
        "_model",
        "_model_id",
        "_normalize",
        "_torch",
    )

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        torch_num_threads: int = 1,
        normalize: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        dim: int | None = None,
    ) -> None:
        """Load the pinned model under the offline environment.

        Args:
            model_id: The pinned sentence-transformers model id (provenance).
            revision: Pinned model revision for reproducibility.
            device: Compute device for :meth:`encode` — ``"auto"`` (default)
                detects CUDA, then Apple MPS, then falls back to CPU; an explicit
                ``"cpu"``/``"mps"``/``"cuda"`` pins that device. ``"auto"`` is
                itself overridable via the ``REDSTACK_OFFLINE_DEVICE`` env var.
                :meth:`export_onnx` always traces on CPU regardless of this value
                (Adapters §4) — the device choice only affects encode throughput,
                never the exported artifact's correctness.
            torch_num_threads: Pinned torch thread count (CPU path only).
            normalize: Apply L2 normalization to outputs (fixed contract).
            batch_size: Default encode batch size.
            dim: Optional dimensionality override; otherwise read from the model.

        Raises:
            RuntimeError: an online marker is set.
            EmbeddingError: the model could not be loaded, or an explicit
                ``REDSTACK_OFFLINE_DEVICE`` override names an unknown device.
        """
        _guard_offline_only()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            torch_module = cast("_Torch", importlib.import_module("torch"))
            resolved_device = (
                _select_device(torch_module) if device == "auto" else device
            )
            torch_module.set_num_threads(torch_num_threads)
            st_module = importlib.import_module("sentence_transformers")
            transformer_cls = st_module.SentenceTransformer
            loaded = transformer_cls(
                model_id, revision=revision, device=resolved_device
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise EmbeddingError(
                f"cannot load sentence-transformers model {model_id!r}: {exc}"
            ) from exc

        model = cast("_StModel", loaded)
        self._torch: Final[_Torch] = torch_module
        self._model: Final[_StModel] = model
        self._model_id: Final[str] = model_id
        self._normalize: Final[bool] = normalize
        self._batch_size: Final[int] = batch_size
        self._device: Final[str] = resolved_device
        self._dim: Final[int] = (
            dim if dim is not None else int(model.get_sentence_embedding_dimension())
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

    @property
    def device(self) -> str:
        """The resolved compute device used by ``encode`` (``cpu``/``mps``/``cuda``).

        Build provenance only (Adapters §4 device policy) — ``export_onnx``
        always traces on CPU regardless of this value.
        """
        return self._device

    @property
    def opset(self) -> int:
        """The pinned ONNX opset :meth:`export_onnx` exports at by default.

        Required by the ``OnnxExportCapable`` Protocol (a ``runtime_checkable``
        Protocol checks attribute *presence*, not signature — without this
        property ``isinstance(adapter, OnnxExportCapable)`` is ``False`` even
        though :meth:`export_onnx` itself is fully implemented).
        """
        return _DEFAULT_OPSET

    @property
    def tokenizer_json(self) -> str:
        """The fast tokenizer's ``tokenizers.Tokenizer.from_str`` JSON payload.

        The online ``OnnxEmbeddingModelAdapter`` fallback encoder tokenizes
        through this exact serialization, so it must travel as its own
        artifact alongside ``model/encoder.onnx`` (Adapters §4).
        """
        return self._model.tokenizer.backend_tokenizer.to_str()

    def encode(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> FloatMatrix:
        """Encode pre-composed documents into a read-only ``(len(texts), dim)`` matrix.

        Output is ``float32``, each row L2-normalized within epsilon, row order
        equal to input order regardless of batching.

        Raises:
            EmbeddingError: the encode operation failed.
        """
        n = len(texts)
        if n == 0:
            empty = np.empty((0, self._dim), dtype=np.float32)
            empty.flags.writeable = False
            return empty

        step = batch_size if batch_size is not None and batch_size > 0 else self._batch_size
        try:
            raw = self._model.encode(
                list(texts),
                batch_size=step,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize,
            )
        except Exception as exc:  # the library raises bare exceptions
            raise EmbeddingError(f"sentence-transformers encode failed: {exc}") from exc

        matrix = np.asarray(raw, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (n, self._dim):
            raise EmbeddingError(
                f"encoded shape {matrix.shape} != expected {(n, self._dim)}"
            )
        matrix.flags.writeable = False
        return matrix

    # ------------------------------------------------------------------ #
    # ONNX export + parity (offline build responsibility, Adapters §4).
    # ------------------------------------------------------------------ #
    def export_onnx(
        self,
        output_path: Path,
        *,
        opset: int = _DEFAULT_OPSET,
        sample_texts: Sequence[str] | None = None,
        max_seq_length: int = 256,
    ) -> float:
        """Export the transformer twin to ``output_path`` and verify st↔onnx parity.

        Exports the underlying HuggingFace transformer (token-embedding output);
        the online onnx adapter applies the matching mean pooling + L2 norm. A
        sample is encoded through both paths and the mean cosine is asserted
        ``>= 0.999``.

        Args:
            output_path: Destination ``.onnx`` path.
            opset: Pinned ONNX opset version.
            sample_texts: Texts for the parity check (a small default if omitted).
            max_seq_length: Tokenization truncation length for the export sample.

        Returns:
            The mean cosine similarity between st and onnx sentence embeddings.

        Raises:
            EmbeddingError: export failed or parity fell below the floor.
        """
        samples = list(sample_texts) if sample_texts else ["the quick brown fox", "hello world"]
        try:
            transformer = cast("_Pooling", self._model[0]).auto_model
            tokenizer = self._model.tokenizer
            tokens = tokenizer(
                samples,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # torch.onnx.export traces most reliably off a CPU-resident model
            # (the tokenizer's "pt" tensors are already CPU); this is a fixed
            # small-sample trace, not the encode throughput path, so it is
            # always pinned to CPU regardless of self._device and restored
            # afterward so a later encode() still runs on the configured device.
            transformer.to("cpu")
            try:
                self._torch.onnx.export(
                    transformer,
                    (input_ids, attention_mask),
                    str(output_path),
                    input_names=["input_ids", "attention_mask"],
                    output_names=["last_hidden_state"],
                    dynamic_axes={
                        "input_ids": {0: "batch", 1: "sequence"},
                        "attention_mask": {0: "batch", 1: "sequence"},
                        "last_hidden_state": {0: "batch", 1: "sequence"},
                    },
                    opset_version=opset,
                    do_constant_folding=True,
                    # Force the legacy TorchScript-based exporter: the dynamic_axes
                    # kwarg above is that exporter's API, and the newer dynamo=True
                    # default (torch>=2.5) requires an additional `onnxscript`
                    # dependency this offline build does not declare.
                    dynamo=False,
                )
            finally:
                transformer.to(self._device)
        except Exception as exc:  # torch/transformers raise bare exceptions
            raise EmbeddingError(f"onnx export failed: {exc}") from exc

        parity = self._parity_cosine(output_path, samples, max_seq_length)
        if parity < _PARITY_FLOOR:
            raise EmbeddingError(
                f"st<->onnx parity {parity:.6f} below floor {_PARITY_FLOOR}"
            )
        return parity

    def _parity_cosine(
        self, onnx_path: Path, samples: Sequence[str], max_seq_length: int
    ) -> float:
        """Mean cosine between st embeddings and onnx-pooled embeddings."""
        try:
            ort_module = importlib.import_module("onnxruntime")
            session = cast(
                "_OrtSession",
                ort_module.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                ),
            )
            tokenizer = self._model.tokenizer
            tokens = tokenizer(
                list(samples),
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="np",
            )
            input_ids = np.asarray(
                cast("npt.NDArray[np.int64]", tokens["input_ids"]), dtype=np.int64
            )
            attention_mask = np.asarray(
                cast("npt.NDArray[np.int64]", tokens["attention_mask"]), dtype=np.int64
            )
            raw = session.run(
                ["last_hidden_state"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
        except Exception as exc:
            raise EmbeddingError(f"onnx parity inference failed: {exc}") from exc

        token_output = np.asarray(raw[0], dtype=np.float32)
        mask = attention_mask.astype(np.float32)[:, :, None]
        summed = np.sum(token_output * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        onnx_vecs = self._unit_rows((summed / counts).astype(np.float32, copy=False))

        st_vecs = self._unit_rows(np.asarray(self.encode(list(samples)), dtype=np.float32))
        cosines = np.sum(onnx_vecs * st_vecs, axis=1)
        return float(np.mean(cosines))

    @staticmethod
    def _unit_rows(matrix: FloatMatrix) -> FloatMatrix:
        norms = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), a_min=1e-12, a_max=None)
        unit: FloatMatrix = (matrix / norms).astype(np.float32, copy=False)
        return unit


if TYPE_CHECKING:
    from redstack.ports.embedding import DeviceReporting, EmbeddingModelPort

    # Compile-time structural conformance to the frozen port surface.
    _PORT_CONFORMANCE: type[EmbeddingModelPort] = SentenceTransformerEmbeddingAdapter
    _DEVICE_CONFORMANCE: type[DeviceReporting] = SentenceTransformerEmbeddingAdapter


__all__: tuple[str, ...] = ("SentenceTransformerEmbeddingAdapter",)