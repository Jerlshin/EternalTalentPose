

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, cast

import typer
import yaml

from redstack.config.determinism import apply_determinism
from redstack.config.loader import ConfigLoadError, load_config
from redstack.config.schema import Profile, RunMode
from redstack.observability.timing import format_duration

__all__: tuple[str, ...] = ("build",)

#: Default online ``ScoringPolicy.neutral_prior`` (config/schema.py), reused
#: when packaging the locked-heuristics scoring_weights artifact.
_DEFAULT_NEUTRAL_PRIOR: float = 0.5


def _load_seed_weights(configs_root: Path) -> tuple[dict[str, float], float]:
    """Read ``weights/scoring_weights.yaml`` -> (component weights, neutral prior).

    This is the locked-heuristics default: the human-authored seed weights,
    used verbatim (no gold-label search) when ``--golden-labels`` is not given.
    """
    path = configs_root / "weights" / "scoring_weights.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    weights = {str(k): float(v) for k, v in doc["weights"].items()}
    neutral_prior = float(doc.get("neutral_prior", _DEFAULT_NEUTRAL_PRIOR))
    return weights, neutral_prior


def _resolve_configs_root(config_arg: Path) -> Path:
    """Resolve a ``--config`` argument (file or directory) to the configs root.

    Accepts the root directory itself, ``<root>/base.yaml``, or
    ``<root>/runtime/<mode>.yaml`` — every form used by the Makefile / CLAUDE.md
    invocations — by walking up to find the directory holding ``base.yaml``.

    Raises:
        typer.BadParameter: no ``base.yaml`` found within two levels up.
    """
    start = config_arg if config_arg.is_dir() else config_arg.parent
    for probe in (start, start.parent):
        if (probe / "base.yaml").is_file():
            return probe
    raise typer.BadParameter(
        f"cannot locate a configs root (base.yaml) from {config_arg}"
    )


def _code_version() -> str:
    """Return the build's code provenance for the report (package version)."""
    import redstack

    return redstack.__version__


def build(
    config: Path = typer.Option(
        Path("configs"),
        "--config",
        help="Configs root directory, or a YAML file inside it.",
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Optional override layer: ci | local."
    ),
    force: str | None = typer.Option(
        None,
        "--force",
        help="Comma-separated stage ids to force-recompute (e.g. 'O9,O15').",
    ),
    golden_labels: bool = typer.Option(
        True,
        "--golden-labels/--no-golden-labels",
        help=(
            "Run the real O8-O10 gold-label calibration search against the "
            "committed data/golden/golden_labels.csv (default — fails loudly "
            "via GoldLabelSeedMissingError if it isn't committed). Pass "
            "--no-golden-labels to skip O8-O10's search and use the locked "
            "heuristic weights from configs/weights/scoring_weights.yaml "
            "as-is; this is an uncalibrated dev shortcut, not a substitute "
            "for the real search."
        ),
    ),
) -> None:
    """Run the offline O0-O18 build: artifacts/ + MANIFEST.json."""
    cumulative_started = time.perf_counter()
    configs_root = _resolve_configs_root(config)
    profile_enum = Profile(profile) if profile else None
    try:
        resolved = load_config(configs_root, RunMode.OFFLINE, profile_enum)
    except ConfigLoadError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    apply_determinism(resolved.determinism)

    # Dynamic on purpose — see module docstring (layering seam + import order).
    compose = importlib.import_module("redstack.pipelines.offline.compose")

    force_ids = tuple(s.strip() for s in force.split(",")) if force else None
    pass_started = time.perf_counter()
    if golden_labels:
        run_offline_build = cast("Any", compose.run_offline_build)
        report = run_offline_build(
            resolved,
            configs_root=configs_root,
            code_version=_code_version(),
            force=force_ids,
        )
    else:
        component_weights, neutral_prior = _load_seed_weights(configs_root)
        run_offline_build_with_locked_heuristics = cast(
            "Any", compose.run_offline_build_with_locked_heuristics
        )
        report = run_offline_build_with_locked_heuristics(
            resolved,
            configs_root=configs_root,
            code_version=_code_version(),
            component_weights=component_weights,
            neutral_prior=neutral_prior,
            force=force_ids,
        )
    pass_elapsed = time.perf_counter() - pass_started
    cumulative_elapsed = time.perf_counter() - cumulative_started

    typer.echo(
        "offline build complete: "
        f"executed={len(report.executed)} "
        f"skipped={len(report.skipped)} "
        f"failed={len(report.failed)}"
    )
    typer.echo(f"offline pass duration:  {format_duration(pass_elapsed)} (MM:SS.ms)")
    typer.echo(
        f"cumulative wall-clock:  {format_duration(cumulative_elapsed)} (MM:SS.ms)"
    )
    if report.failed:
        raise typer.Exit(code=1)
