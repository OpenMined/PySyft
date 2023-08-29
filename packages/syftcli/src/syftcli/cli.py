# third party
from rich import box
from rich.console import Console
from rich.table import Table
from syftcli.version import __version__
import typer

app = typer.Typer(name="Syft CLI", no_args_is_help=True)
console = Console()


@app.command()
def hello() -> None:
    typer.echo("Syft CLI 👋")


@app.command()
def version() -> None:
    table = Table(box=box.HORIZONTALS)
    table.add_column("Package", no_wrap=True)
    table.add_column("Version", style="cyan")

    table.add_row("CLI", __version__)
    table.add_row("Library", get_syft_version())

    console.print(table)


def get_syft_version() -> str:
    try:
        # syft absolute
        import syft

        return syft.__version__
    except ModuleNotFoundError:
        return "Not Installed."


if __name__ == "__main__":
    app()
