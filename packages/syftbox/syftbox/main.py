from pathlib import Path
from typing import Annotated, Optional

from rich import print as rprint
from typer import Exit, Option, Typer

from syftbox import __version__
from syftbox.app.cli import app as app_cli
from syftbox.client.cli import app as client_cli
from syftbox.server.cli import app as server_cli
from syftbox.tui.cli import app as tui_cli

app = Typer(
    name="SyftBox",
    help="SyftBox CLI",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

CONFIG_OPTS = Option("-c", "--config", "--config_path", help="Path to the SyftBox config")


@app.command(rich_help_panel="General Options")
def version() -> None:
    """Print SyftBox version"""

    print(__version__)


@app.command(rich_help_panel="General Options")
def debug(config_path: Annotated[Optional[Path], CONFIG_OPTS] = None) -> None:
    """Print SyftBox debug data"""

    # lazy import to improve CLI startup performance
    from syftbox.lib.debug import debug_report_yaml

    try:
        rprint(debug_report_yaml(config_path))
    except Exception as e:
        rprint(f"[red]Error[/red]: {e}")
        raise Exit(1)


app.add_typer(client_cli, name="client")
app.add_typer(server_cli, name="server")
app.add_typer(app_cli, name="app")
app.add_typer(tui_cli, name="tui")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
