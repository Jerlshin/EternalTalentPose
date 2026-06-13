"""The single home for determinism policy (Repository Layout §11, Architecture §8).

Determinism is *configuration*, owned in exactly one file and asserted at
startup. This module pins BLAS/OpenMP threads, constructs the seeded NumPy
generator, and builds the CPU-only ONNX Runtime ``SessionOptions``. No
``datetime.now`` and no online RNG live anywhere reachable from the online
pipeline; the clock is the injected ``as_of`` and ties resolve by ascending
``candidate_id``.

**Thread-pin ordering contract.** BLAS/OpenMP read their thread caps from the
environment *at library import time*. :func:`apply_determinism` must therefore
be called by the CLI/composition root **before** any NumPy/ONNX-importing module
is imported. To preserve that window, this module imports NumPy and ONNX Runtime
only lazily (inside functions), so importing :mod:`redstack.config.determinism`
itself never pulls a math runtime into the process.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from redstack.config.schema import DeterminismConfig

if TYPE_CHECKING:  # import-time-free typing only; no runtime math-runtime import.
    import numpy as np
    import onnxruntime as ort

__all__ = [
    "CPU_EXECUTION_PROVIDER",
    "THREAD_ENV_VARS",
    "apply_determinism",
    "assert_determinism",
    "make_rng",
    "build_onnx_session_options",
]

#: The only ONNX Runtime execution provider permitted online (CPU-only sandbox).
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"

#: Environment variables that cap native threading for every BLAS backend we
#: may encounter. ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` are authoritative
#: from config; the rest mirror the OMP cap to prevent oversubscription thrash.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _expected_env(config: DeterminismConfig) -> dict[str, str]:
    """The exact thread-cap environment this config mandates."""
    omp = str(config.omp_num_threads)
    return {
        "OMP_NUM_THREADS": omp,
        "MKL_NUM_THREADS": str(config.mkl_num_threads),
        "OPENBLAS_NUM_THREADS": omp,
        "NUMEXPR_NUM_THREADS": omp,
        "VECLIB_MAXIMUM_THREADS": omp,
    }


def apply_determinism(config: DeterminismConfig) -> None:
    """Pin native thread counts for thread-count-invariant output.

    Sets the BLAS/OpenMP thread-cap environment variables from ``config``. Must
    be invoked at process start, before NumPy or ONNX Runtime are imported, so
    the caps take effect when those libraries initialize.

    This function performs no other global mutation: the NumPy generator and
    ONNX session options are returned by value from :func:`make_rng` and
    :func:`build_onnx_session_options`, never installed as global state.
    """
    for key, value in _expected_env(config).items():
        os.environ[key] = value


def assert_determinism(config: DeterminismConfig) -> None:
    """Verify the thread pins are in effect; raise if they are not.

    Called at startup immediately after :func:`apply_determinism` (and after the
    math runtimes are imported) to guarantee the run cannot proceed with an
    unpinned, non-reproducible thread configuration.

    Raises:
        RuntimeError: If any mandated thread-cap variable is missing or differs
            from the configured value.
    """
    expected = _expected_env(config)
    mismatched: list[str] = []
    for key, want in expected.items():
        have = os.environ.get(key)
        if have != want:
            mismatched.append(f"{key}: expected {want!r}, got {have!r}")
    if mismatched:
        detail = "; ".join(mismatched)
        raise RuntimeError(
            "determinism assertion failed — thread pins not in effect: " + detail
        )


def make_rng(config: DeterminismConfig) -> np.random.Generator:
    """Construct the single seeded NumPy generator for offline RNG.

    Online RNG is disabled by policy (ties break by ``candidate_id``); this
    generator is used only by the offline pipeline (O7 archetypes, O8 labeling)
    via the entropy port. Uses ``default_rng`` (PCG64) for a reproducible,
    explicitly-seeded stream.
    """
    import numpy as np

    return np.random.default_rng(config.seed)


def build_onnx_session_options(config: DeterminismConfig) -> ort.SessionOptions:
    """Build CPU-only, single-thread-pinned ONNX Runtime session options.

    Pins intra-/inter-op thread counts, forces sequential execution, and leaves
    the caller to register only :data:`CPU_EXECUTION_PROVIDER` — the online
    sandbox is CPU-only and network-isolated. The returned options carry no
    global state.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = config.onnx_intra_op_threads
    options.inter_op_num_threads = config.onnx_inter_op_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options
