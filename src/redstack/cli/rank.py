

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import typer

from redstack.config.determinism import apply_determinism
from redstack.config.loader import ConfigLoadError, load_config
from redstack.config.schema import Profile, RunMode

__all__: tuple[str, ...] = ("rank",)


def _resolve_configs_root(config_arg: Path) -> Path:
    """Resolve a ``--config`` argument (file or directory) to the configs root.

    See :func:`redstack.cli.build._resolve_configs_root` for the same logic.
    """
    start = config_arg if config_arg.is_dir() else config_arg.parent
    for probe in (start, start.parent):
        if (probe / "base.yaml").is_file():
            return probe
    raise typer.BadParameter(
        f"cannot locate a configs root (base.yaml) from {config_arg}"
    )


def rank(
    input: Path = typer.Option(  # noqa: A002 — mirrors the documented --input flag.
        Path("data/raw/candidates.jsonl"),
        "--input",
        help="Candidates .jsonl to rank.",
    ),
    output: Path = typer.Option(
        Path("artifacts/submission.csv"), "--output", help="Destination submission.csv."
    ),
    config: Path = typer.Option(
        Path("configs"),
        "--config",
        help="Configs root directory, or a YAML file inside it.",
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Optional override layer: ci | local."
    ),
    participant_id: str = typer.Option(
        "submission", "--participant-id", help="Output filename stem for the report."
    ),
) -> None:
    """Run the online R0-R9 ranking pass: submission.csv + run_report.json."""
    configs_root = _resolve_configs_root(config)
    profile_enum = Profile(profile) if profile else None
    try:
        resolved = load_config(configs_root, RunMode.ONLINE, profile_enum)
    except ConfigLoadError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    apply_determinism(resolved.determinism)

    # Dynamic on purpose — see module docstring (layering seam + import order).
    compose = importlib.import_module("redstack.pipelines.online.compose")
    run_online_rank = cast("Any", compose.run_online_rank)

    output = output.resolve()
    report_path = output.with_name("run_report.json")
    result = run_online_rank(
        resolved,
        input_path=input.resolve(),
        output_path=output,
        report_path=report_path,
        participant_id=participant_id,
    )

    typer.echo(
        "online rank complete: "
        f"rows={result.row_count} "
        f"honeypot_rate_top100={result.honeypot_rate_top100:.4f} "
        f"within_budget={result.within_budget} "
        f"peak_rss_mb={result.peak_rss_mb:.1f}"
    )
    if not result.within_budget:
        raise typer.Exit(code=1)
