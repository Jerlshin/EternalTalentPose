

from __future__ import annotations

import typer

from redstack.cli.build import build
from redstack.cli.rank import rank
from redstack.cli.validate import validate

__all__: tuple[str, ...] = ("app",)

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.command(name="build")(build)
app.command(name="rank")(rank)
app.command(name="validate")(validate)


if __name__ == "__main__":
    app()
