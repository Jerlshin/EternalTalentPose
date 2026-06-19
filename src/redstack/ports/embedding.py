
from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable
from redstack.ports._types import FloatMatrix
from redstack.domain.errors import DomainError


class EmbeddingError(DomainError):
    """Encode/export runtime failure in an embedding adapter (Ports §1)."""


class EmbeddingModelPort(Protocol):
    @property
    def dim(self) -> int: ...
    @property
    def model_id(self) -> str: ...
    def encode(self, texts: Sequence[str], *, batch_size: int | None = None) -> FloatMatrix: ...


@runtime_checkable
class OnnxExportCapable(Protocol):
    """Offline-only ONNX export + parity capability (Adapters §4).

    Implemented by the sentence-transformers offline adapter; absent on the
    online onnx adapter. ``export_onnx`` writes ``encoder.onnx`` to ``dest`` at a
    pinned opset and returns the st<->onnx parity cosine on a sample (target
    >= 0.999; the adapter raises ``EmbeddingError`` if parity fails so a bad twin
    is never accepted). ``tokenizer_json`` serializes the same fast tokenizer the
    onnx twin was traced against, as the online ``tokenizers.Tokenizer.from_str``
    payload — the online onnx fallback encoder cannot tokenize without it.
    """

    @property
    def opset(self) -> int: ...
    @property
    def tokenizer_json(self) -> str: ...
    def export_onnx(self, dest: Path, *, sample_texts: Sequence[str]) -> float: ...