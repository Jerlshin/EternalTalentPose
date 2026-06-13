"""Deterministic configuration composition (Repository Layout §11).

This module performs the only IO in the config layer: reading YAML files. The
merge itself is **pure** — a deterministic, order-fixed deep-merge of
``base.yaml -> runtime/<mode>.yaml -> profiles/<profile>.yaml`` — followed by a
single ``validate-or-die`` into the typed :class:`RedstackConfig`. Untyped
config never escapes this module; callers always receive a frozen Pydantic VO.

Import boundary: this module reads files and so may be imported **only** by
``pipelines`` and ``cli``; ``engines``/``features``/``domain`` see only the pure
:mod:`redstack.config.schema`.

The behaviour/authoring seeds (``weights``/``lexicon``/``anchors``/``gates``/
``integrity``) are loaded individually by the offline pipeline stages; the
loaders for them live here too so that all file-reading config IO has a single
home.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from redstack.config.schema import (
    EligibilityRulesConfig,
    HoneypotRulesConfig,
    JdAnchorsConfig,
    LexiconSeedConfig,
    Profile,
    RedstackConfig,
    RunMode,
    ScoringWeightsConfig,
)

__all__ = [
    "ConfigLoadError",
    "load_config",
    "load_scoring_weights",
    "load_lexicon_seed",
    "load_jd_anchors",
    "load_eligibility_rules",
    "load_honeypot_rules",
    "config_fingerprint",
    "deep_merge",
]

_ModelT = TypeVar("_ModelT", bound=BaseModel)

# Fixed layer filenames within the configs root. Order is load-bearing.
_BASE_FILENAME = "base.yaml"
_RUNTIME_DIR = "runtime"
_PROFILES_DIR = "profiles"

# Behaviour / authoring seed locations (relative to the configs root).
_SCORING_WEIGHTS = ("weights", "scoring_weights.yaml")
_LEXICON_SEED = ("lexicon", "lexicon.seed.yaml")
_JD_ANCHORS = ("anchors", "jd_anchors.yaml")
_ELIGIBILITY_RULES = ("gates", "eligibility_rules.yaml")
_HONEYPOT_RULES = ("integrity", "honeypot_rules.yaml")


class ConfigLoadError(RuntimeError):
    """Raised when a config file is missing, malformed, or fails validation.

    Carries a human-readable, deterministic message; never leaks a partially
    constructed config object.
    """


# --------------------------------------------------------------------------- #
# Pure merge.                                                                  #
# --------------------------------------------------------------------------- #
def deep_merge(
    base: Mapping[str, object], override: Mapping[str, object]
) -> dict[str, object]:
    """Deterministically deep-merge ``override`` over ``base``.

    Rules (fixed, total, side-effect-free):

    * Two mappings at the same key are merged recursively.
    * Any other value in ``override`` replaces the ``base`` value wholesale
      (including lists — sequences are replaced, never concatenated, so the
      result is a pure function of the inputs).
    * Keys absent from ``override`` are carried through unchanged.

    The output dict preserves ``base`` key order followed by override-only keys,
    making the merge byte-stable for a fixed input pair.
    """
    merged: dict[str, object] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


# --------------------------------------------------------------------------- #
# IO helpers.                                                                  #
# --------------------------------------------------------------------------- #
def _read_yaml_mapping(path: Path) -> dict[str, object]:
    """Read a YAML file into a string-keyed mapping, or fail loudly.

    An empty file is treated as an empty mapping (a valid override layer). Any
    non-mapping top-level document is an authoring error.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(f"cannot read config file: {path}") from exc

    try:
        document = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"invalid YAML in {path}: {exc}") from exc

    if document is None:
        return {}
    if not isinstance(document, Mapping):
        raise ConfigLoadError(
            f"config file {path} must contain a mapping at the top level"
        )
    return {str(key): value for key, value in document.items()}


def _validate_into(model: type[_ModelT], data: Mapping[str, object], origin: str) -> _ModelT:
    """Validate ``data`` into ``model`` or raise :class:`ConfigLoadError`."""
    try:
        return model.model_validate(dict(data))
    except ValidationError as exc:
        raise ConfigLoadError(f"config validation failed for {origin}:\n{exc}") from exc


def _read_seed(
    configs_root: Path, parts: tuple[str, str], model: type[_ModelT]
) -> _ModelT:
    """Read and validate a single behaviour/authoring seed file."""
    path = configs_root.joinpath(*parts)
    data = _read_yaml_mapping(path)
    return _validate_into(model, data, origin=str(path))


# --------------------------------------------------------------------------- #
# Public composition entrypoint.                                              #
# --------------------------------------------------------------------------- #
def load_config(
    configs_root: Path,
    run_mode: RunMode,
    profile: Profile | None = None,
) -> RedstackConfig:
    """Compose and validate the runtime configuration.

    Reads ``base.yaml``, ``runtime/<run_mode>.yaml`` and, when ``profile`` is
    given, ``profiles/<profile>.yaml``; deep-merges them in that fixed order;
    injects the authoritative ``run_mode``/``profile`` keys (so YAML never
    declares them and cannot drift); and validates the result into a frozen
    :class:`RedstackConfig`.

    Args:
        configs_root: Path to the ``configs/`` directory.
        run_mode: Which ``runtime/<mode>.yaml`` layer to compose.
        profile: Optional final override layer.

    Returns:
        The fully-validated, frozen :class:`RedstackConfig`.

    Raises:
        ConfigLoadError: If any layer is missing, malformed, or the composed
            config fails schema validation.
    """
    base = _read_yaml_mapping(configs_root / _BASE_FILENAME)
    runtime = _read_yaml_mapping(
        configs_root / _RUNTIME_DIR / f"{run_mode.value}.yaml"
    )

    merged = deep_merge(base, runtime)
    if profile is not None:
        profile_layer = _read_yaml_mapping(
            configs_root / _PROFILES_DIR / f"{profile.value}.yaml"
        )
        merged = deep_merge(merged, profile_layer)

    # The loader is the single authority for these identity keys.
    merged["run_mode"] = run_mode.value
    merged["profile"] = profile.value if profile is not None else None

    # Profiles are mode-agnostic and may carry overrides for both modes; the
    # loader keeps only the active run-mode's block so the inactive one cannot
    # trip the mode-consistency invariant on RedstackConfig.
    inactive = RunMode.OFFLINE if run_mode is RunMode.ONLINE else RunMode.ONLINE
    merged.pop(inactive.value, None)

    return _validate_into(
        RedstackConfig,
        merged,
        origin=f"{configs_root} (mode={run_mode.value}, profile={profile})",
    )


# --------------------------------------------------------------------------- #
# Behaviour / authoring seed loaders (offline pipeline).                      #
# --------------------------------------------------------------------------- #
def load_scoring_weights(configs_root: Path) -> ScoringWeightsConfig:
    """Load the O9 candidate scoring-weight seed."""
    return _read_seed(configs_root, _SCORING_WEIGHTS, ScoringWeightsConfig)


def load_lexicon_seed(configs_root: Path) -> LexiconSeedConfig:
    """Load the O4 lexicon seed terms."""
    return _read_seed(configs_root, _LEXICON_SEED, LexiconSeedConfig)


def load_jd_anchors(configs_root: Path) -> JdAnchorsConfig:
    """Load the O6 JD anchor intents."""
    return _read_seed(configs_root, _JD_ANCHORS, JdAnchorsConfig)


def load_eligibility_rules(configs_root: Path) -> EligibilityRulesConfig:
    """Load the O6 JD eligibility rule seed."""
    return _read_seed(configs_root, _ELIGIBILITY_RULES, EligibilityRulesConfig)


def load_honeypot_rules(configs_root: Path) -> HoneypotRulesConfig:
    """Load the O3 honeypot rule-shape seed."""
    return _read_seed(configs_root, _HONEYPOT_RULES, HoneypotRulesConfig)


# --------------------------------------------------------------------------- #
# Reproducibility.                                                            #
# --------------------------------------------------------------------------- #
def config_fingerprint(config: RedstackConfig) -> str:
    """Return the deterministic sha256 hex digest of a resolved config.

    Serializes the config to canonical JSON (enums by value, sorted keys, no
    insignificant whitespace) and hashes the UTF-8 bytes. This is the
    ``config_hash`` recorded into the run report's ``reproducible`` block;
    identical inputs ⇒ identical digest, independent of dict iteration order.
    """
    payload = config.model_dump(mode="json")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
