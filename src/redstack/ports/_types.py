"""REDSTACK Ports Layer — Shared non-domain architectural abstractions and data containers."""
import numpy as np
import numpy.typing as npt

__all__ = ["FloatVector", "FloatMatrix", "SourceRecord"]

FloatVector = npt.NDArray[np.float32]
FloatMatrix = npt.NDArray[np.float32]
