# third party
from rich import box
from rich.console import Console
from rich.table import Table
import typer
from version import __version__

app = typer.Typer(name="Syft CLI", no_args_is_help=True)
console = Console()


@app.command()
def hello():
    typer.echo("Syft CLI 👋")


@app.command()
def version():
    # syft absolute
    import syft

    table = Table(box=box.HORIZONTALS)
    table.add_column("Package", no_wrap=True)
    table.add_column("Version", style="cyan")

    table.add_row("CLI", __version__)
    table.add_row("Library", syft.__version__)

    console.print(table)
