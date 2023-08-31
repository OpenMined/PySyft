# third party
from rich import box
from rich.table import Table
from typer import Typer
from typer import echo

# syftcli absolute
from syftcli.bundle import cmd as bundle_cmd
from syftcli.core.console import console
from syftcli.version import __version__

__all__ = "app"

app = Typer(name="Syft CLI", no_args_is_help=True, pretty_exceptions_show_locals=False)
app.add_typer(bundle_cmd, name="bundle")


@app.command()
def hello() -> None:
    echo("Syft CLI 👋")


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
        return "Not Installed"


if __name__ == "__main__":
    app()
