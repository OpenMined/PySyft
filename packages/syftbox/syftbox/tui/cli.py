from pathlib import Path
from typing import Annotated
from venv import logger

from rich import print as rprint
from typer import Exit, Option, Typer

from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.client_shim import Client
from syftbox.lib.constants import DEFAULT_CONFIG_PATH
from syftbox.lib.exceptions import ClientConfigException

app = Typer(
    name="SyftBox Terminal UI",
    help="Launch the SyftBox Terminal UI",
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

CONFIG_OPTS = Option("-c", "--config", "--config_path", help="Path to the SyftBox config")


@app.callback(invoke_without_command=True)
def run_tui(
    config_path: Annotated[Path, CONFIG_OPTS] = DEFAULT_CONFIG_PATH,
) -> None:
    # Late import to avoid long startup times
    from syftbox.tui.app import SyftBoxTUI

    syftbox_context = get_syftbox_context(config_path)
    tui = SyftBoxTUI(syftbox_context)
    logger.debug("Running SyftBox TUI")
    tui.run()


def get_syftbox_context(config_path: Path) -> Client:
    try:
        conf = SyftClientConfig.load(config_path)
        context = Client(conf)
        return context
    except ClientConfigException:
        msg = (
            f"[bold red]Error:[/bold red] Couldn't load config at: [yellow]'{config_path}'[/yellow]\n"
            "Please ensure that:\n"
            "  - The configuration file exists at the specified path.\n"
            "  - You've run the SyftBox atleast once.\n"
            f"  - For custom configs, provide the proper path using [cyan]--config[/cyan] flag"
        )
        rprint(msg)
        raise Exit(1)
    except Exception as e:
        rprint(f"[bold red]Error:[/bold red] {e}")
        raise Exit(1)


if __name__ == "__main__":
    app()
