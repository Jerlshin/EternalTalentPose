

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from redstack.config.determinism import apply_determinism
from redstack.config.loader import ConfigLoadError, load_config
from redstack.config.schema import Profile, RunMode
from redstack.observability.timing import format_duration

if TYPE_CHECKING:
    from redstack.pipelines.online.pipeline import OnlinePipelineResult

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
    cumulative_started = time.perf_counter()
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
    pass_started = time.perf_counter()

    def _report_and_exit(result: OnlinePipelineResult) -> None:
        """Print the terminal status, then hard-exit before teardown.

        At full 100k-candidate scale, letting this process return normally
        through `run_online_rank` -> `OnlinePipeline.run` means CPython has to
        refcount-deallocate tens of millions of live small objects still
        referenced by that call's locals — measured at ~165s wall-clock, more
        than any single R-stage. `submission.csv` and `run_report.json` are
        already durably written by this point (R8/R9), so that teardown buys
        nothing; skip it with `os._exit`, which lets the OS reclaim the whole
        process's memory in one step instead. `os._exit` bypasses normal
        stdio flushing, so flush explicitly first.
        """
        pass_elapsed = time.perf_counter() - pass_started
        cumulative_elapsed = time.perf_counter() - cumulative_started
        typer.echo(
            "online rank complete: "
            f"rows={result.row_count} "
            f"honeypot_rate_top100={result.honeypot_rate_top100:.4f} "
            f"within_budget={result.within_budget} "
            f"peak_rss_mb={result.peak_rss_mb:.1f}"
        )
        typer.echo(
            f"online pass duration:   {format_duration(pass_elapsed)} (MM:SS.ms)"
        )
        typer.echo(
            f"cumulative wall-clock:  {format_duration(cumulative_elapsed)} (MM:SS.ms)"
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if result.within_budget else 1)

    run_online_rank(
        resolved,
        input_path=input.resolve(),
        output_path=output,
        report_path=report_path,
        participant_id=participant_id,
        on_result=_report_and_exit,
    )
    # Unreachable in practice: `_report_and_exit` always terminates the
    # process. Only reached if a future caller wires a non-exiting
    # `on_result`, so it stays a correct (if redundant) fallback.
    raise AssertionError("run_online_rank returned without invoking on_result")
