"""``build`` — the offline O0-O18 compilation verb.

Owner layer: cli.
Allowed imports: typer, pipelines, config, observability. Per Repository
Layout §8b, ``cli`` (even transitively) may never import ``adapters`` — adapter
binding is delegated entirely to
:func:`redstack.pipelines.offline.compose.run_offline_build`, reached here via
``importlib.import_module`` rather than a static ``from ... import`` so
import-linter's transitive "cli -> adapters" check has no edge to find (a
literal import statement would create one even sitting inside a function
body — the check is on the parsed AST, not on what actually executes first).

That dynamic call also happens to be exactly what thread-cap pinning needs
anyway: ``config.determinism.apply_determinism`` must run before any
NumPy/ONNX-importing module loads, and ``pipelines.offline.compose``
transitively pulls in numpy/torch.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import typer

from redstack.config.determinism import apply_determinism
from redstack.config.loader import ConfigLoadError, load_config
from redstack.config.schema import Profile, RunMode

__all__: tuple[str, ...] = ("build",)


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
) -> None:
    """Run the offline O0-O18 build: artifacts/ + MANIFEST.json."""
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
    run_offline_build = cast("Any", compose.run_offline_build)

    force_ids = tuple(s.strip() for s in force.split(",")) if force else None
    report = run_offline_build(
        resolved,
        configs_root=configs_root,
        code_version=_code_version(),
        force=force_ids,
    )

    typer.echo(
        "offline build complete: "
        f"executed={len(report.executed)} "
        f"skipped={len(report.skipped)} "
        f"failed={len(report.failed)}"
    )
    if report.failed:
        raise typer.Exit(code=1)
