"""EmbeddingModelPort (Ports §1) + the offline-only ONNX-export capability.

The core ``EmbeddingModelPort`` is the frozen text->vector seam (encode + dim +
model_id). ``EmbeddingError`` is the port-level failure type (encode runtime
failure; never returns zeros to mask failure).

``OnnxExportCapable`` is a *narrow optional* secondary Protocol that the offline
``SentenceTransformerEmbeddingAdapter`` (Adapters §4) additionally implements: it
exports the base model's ONNX twin and reports the st<->onnx parity cosine. The
online onnx adapter does NOT implement it. The O13 stage detects this capability
at runtime (``isinstance``) so the frozen ``EmbeddingModelPort`` stays unchanged
while the stage can still register ``model/encoder.onnx`` (Adapters §4: export +
parity verification happen before the artifact is accepted).
"""
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
    is never accepted).
    """

    @property
    def opset(self) -> int: ...
    def export_onnx(self, dest: Path, *, sample_texts: Sequence[str]) -> float: ...