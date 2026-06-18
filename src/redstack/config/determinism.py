"""The single home for determinism policy (Repository Layout §11, Architecture §8).

Determinism is *configuration*, owned in exactly one file and asserted at
startup. This module pins BLAS/OpenMP threads, constructs the seeded NumPy
generator, and builds the CPU-only ONNX Runtime ``SessionOptions``. No
``datetime.now`` and no online RNG live anywhere reachable from the online
pipeline; the clock is the injected ``as_of`` and ties resolve by ascending
``candidate_id``.

**Thread-pin ordering contract.** BLAS/OpenMP read their thread caps from the
environment *at library import time*. :func:`apply_determinism` / :func:`pin_determinism`
must therefore be called by the CLI/composition root **before** any NumPy/ONNX-importing
module is imported. To preserve that window, this module imports NumPy and ONNX Runtime
only lazily (inside functions), so importing :mod:`redstack.config.determinism`
itself never pulls a math runtime into the process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Union, final

from redstack.config.schema import DeterminismConfig

if TYPE_CHECKING:  # import-time-free typing; guards online module load speed
    import numpy as np
    import onnxruntime as ort

__all__ = [
    "CPU_EXECUTION_PROVIDER",
    "THREAD_ENV_VARS",
    "DeterminismPolicy",
    "apply_determinism",
    "assert_determinism",
    "pin_determinism",
    "make_rng",
    "build_onnx_session_options",
    "onnx_session_options",
]

#: The only ONNX Runtime execution provider permitted online (CPU-only sandbox).
CPU_EXECUTION_PROVIDER: Final[str] = "CPUExecutionProvider"

#: Environment variables that cap native threading for every BLAS backend we
#: may encounter. ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` are authoritative
#: from config; the rest mirror the OMP cap to prevent oversubscription thrash.
THREAD_ENV_VARS: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@final
@dataclass(frozen=True, slots=True)
class DeterminismPolicy:
    """The resolved determinism knobs used by online stage variants.

    Maintained for downstream signature compatibility with the online orchestrator passes.
    """

    seed: int
    blas_threads: int = 1
    intra_op_threads: int = 1
    inter_op_threads: int = 1

    def as_dict(self) -> dict[str, int]:
        """Serialize the pinned knobs for the R9 reproducible/audit block."""
        return {
            "seed": self.seed,
            "blas_threads": self.blas_threads,
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
        }


def _get_thread_values(config: DeterminismConfig | DeterminismPolicy) -> tuple[str, str]:
    """Extract thread strings uniformly across config or policy types."""
    if isinstance(config, DeterminismConfig):
        return str(config.omp_num_threads), str(config.mkl_num_threads)
    threads = str(max(1, config.blas_threads))
    return threads, threads


def apply_determinism(config: DeterminismConfig | DeterminismPolicy) -> None:
    """Pin native thread counts for thread-count-invariant output.

    Sets the BLAS/OpenMP thread-cap environment variables from the configuration.
    """
    omp, mkl = _get_thread_values(config)
    os.environ["OMP_NUM_THREADS"] = omp
    os.environ["MKL_NUM_THREADS"] = mkl
    os.environ["OPENBLAS_NUM_THREADS"] = omp
    os.environ["NUMEXPR_NUM_THREADS"] = omp
    os.environ["VECLIB_MAXIMUM_THREADS"] = omp


def assert_determinism(config: DeterminismConfig | DeterminismPolicy) -> None:
    """Verify the thread pins are in effect; raise if they are not.

    Raises:
        RuntimeError: If any mandated thread-cap variable is missing or differs.
    """
    omp, mkl = _get_thread_values(config)
    expected = {
        "OMP_NUM_THREADS": omp,
        "MKL_NUM_THREADS": mkl,
        "OPENBLAS_NUM_THREADS": omp,
        "NUMEXPR_NUM_THREADS": omp,
        "VECLIB_MAXIMUM_THREADS": omp,
    }
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


def pin_determinism(
    policy: Union[DeterminismConfig, DeterminismPolicy]
) -> Union[DeterminismConfig, DeterminismPolicy]:
    """Polymorphic bridge executing both apply and assert passes in one call.

    Matches the exact inline invocation signature expected by the online loop.
    """
    apply_determinism(policy)
    assert_determinism(policy)
    return policy


def make_rng(config: DeterminismConfig | DeterminismPolicy) -> np.random.Generator:
    """Construct the single seeded NumPy generator for offline RNG.

    Used by the offline pipeline (O7 archetypes, O8 labeling) via the entropy port.
    """
    import numpy as np

    return np.random.default_rng(config.seed)


def build_onnx_session_options(config: DeterminismConfig | DeterminismPolicy) -> ort.SessionOptions:
    """Build CPU-only, single-thread-pinned ONNX Runtime session options.

    Forces sequential execution and locks thread counts to prevent float reduction drift.
    """
    import onnxruntime as ort

    if isinstance(config, DeterminismConfig):
        intra = config.onnx_intra_op_threads
        inter = config.onnx_inter_op_threads
    else:
        intra = max(1, config.intra_op_threads)
        inter = max(1, config.inter_op_threads)

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra
    options.inter_op_num_threads = inter
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def onnx_session_options(config: DeterminismConfig | DeterminismPolicy) -> ort.SessionOptions:
    """Alias mapping targeting the online session setup signature."""
    return build_onnx_session_options(config)