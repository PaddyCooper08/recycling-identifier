from ast import Str
import typer
from typing import Optional

app = typer.Typer()


@app.command()
def recycle(
    number: Optional[int] = typer.Argument(None, min=1, max=7), manual: bool = False
):
    if number == 1:
        typer.secho("You can recycle this! :)", fg=typer.colors.GREEN, bold=True)
        typer.secho("1 - PETE - Polyethelene Terephthalate", fg=typer.colors.MAGENTA)
        typer.secho("Recycle kerbside.", fg=typer.colors.BLUE)

    if number == 2:
        typer.secho("You can recycle this! :)", fg=typer.colors.GREEN, bold=True)
        typer.secho("2 – HDPE – High density Polyethylene", fg=typer.colors.MAGENTA)
        typer.secho(
            "Recycle kerbside (but not thin plastics such as plastic wrap)",
            fg=typer.colors.BLUE,
        )

    if number == 3:
        typer.secho("You can't recycle this! :(", fg=typer.colors.RED, bold=True)
        typer.secho("3 – PVC – Polyvinyl Chloride", fg=typer.colors.MAGENTA)

    if number == 4:
        typer.secho("You can't recycle this! :(", fg=typer.colors.RED, bold=True)
        typer.secho("4 – LDPE Low-density Polyethylene", fg=typer.colors.MAGENTA)

    if number == 5:
        typer.secho(
            "You can recycle this! (sometimes) :)", fg=typer.colors.GREEN, bold=True
        )
        typer.secho("5 – PP – Polypropylene", fg=typer.colors.MAGENTA)
        typer.secho(
            "Becoming more readily recycled kerbside - check the guidance in your area.",
            fg=typer.colors.BLUE,
        )

    if number == 6:
        typer.secho("You can't recycle this! :(", fg=typer.colors.RED, bold=True)
        typer.secho("6 – PS – Polystyrene", fg=typer.colors.MAGENTA)

    if number == 7:
        typer.secho("You can't recycle this! :(", fg=typer.colors.RED, bold=True)
        typer.secho("7 – Other", fg=typer.colors.MAGENTA)


@app.command()
def goodbye():
    print("Goodbye")


if __name__ == "__main__":
    app()
