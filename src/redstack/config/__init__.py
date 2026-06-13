"""REDSTACK configuration layer.

Re-exports the pure schema (importable anywhere) and the IO-bearing loader /
determinism policy (importable only by ``pipelines`` and ``cli``). Importing
this package does not import any math runtime; thread pins remain effective when
:func:`apply_determinism` is called before NumPy/ONNX import.
"""

from __future__ import annotations

from redstack.config.determinism import (
    CPU_EXECUTION_PROVIDER,
    THREAD_ENV_VARS,
    apply_determinism,
    assert_determinism,
    build_onnx_session_options,
    make_rng,
)
from redstack.config.loader import (
    ConfigLoadError,
    config_fingerprint,
    deep_merge,
    load_config,
    load_eligibility_rules,
    load_honeypot_rules,
    load_jd_anchors,
    load_lexicon_seed,
    load_scoring_weights,
)
from redstack.config.schema import (
    AnchorIntent,
    AnchorPolarity,
    BudgetConfig,
    ConceptSeed,
    DeterminismConfig,
    EligibilityRule,
    EligibilityRulesConfig,
    HoneypotRule,
    HoneypotRulesConfig,
    HoneypotSeverity,
    JdAnchorsConfig,
    LexiconSeedConfig,
    LogLevel,
    LoggingConfig,
    MalformedRecordPolicy,
    OfflineRuntimeConfig,
    OnlineRuntimeConfig,
    PathsConfig,
    Profile,
    RedstackConfig,
    RunMode,
    ScorePresentationConfig,
    ScoreTransformKind,
    ScoringWeightsConfig,
)

__all__ = [
    # schema — vocabularies
    "RunMode",
    "Profile",
    "LogLevel",
    "MalformedRecordPolicy",
    "ScoreTransformKind",
    "AnchorPolarity",
    "HoneypotSeverity",
    # schema — runtime models
    "DeterminismConfig",
    "BudgetConfig",
    "PathsConfig",
    "LoggingConfig",
    "ScorePresentationConfig",
    "OnlineRuntimeConfig",
    "OfflineRuntimeConfig",
    "RedstackConfig",
    # schema — authoring seeds
    "ScoringWeightsConfig",
    "ConceptSeed",
    "LexiconSeedConfig",
    "AnchorIntent",
    "JdAnchorsConfig",
    "EligibilityRule",
    "EligibilityRulesConfig",
    "HoneypotRule",
    "HoneypotRulesConfig",
    # loader
    "ConfigLoadError",
    "load_config",
    "load_scoring_weights",
    "load_lexicon_seed",
    "load_jd_anchors",
    "load_eligibility_rules",
    "load_honeypot_rules",
    "config_fingerprint",
    "deep_merge",
    # determinism
    "CPU_EXECUTION_PROVIDER",
    "THREAD_ENV_VARS",
    "apply_determinism",
    "assert_determinism",
    "make_rng",
    "build_onnx_session_options",
]
