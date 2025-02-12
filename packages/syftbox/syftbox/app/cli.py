import sys
from pathlib import Path
from typing import List

from loguru import logger
from rich import print as rprint
from typer import Argument, Exit, Option, Typer
from typing_extensions import Annotated

from syftbox import __version__
from syftbox.app.manager import install_app, list_app, uninstall_app
from syftbox.client.base import SyftBoxContextInterface
from syftbox.client.core import SyftBoxRunner
from syftbox.client.plugins.apps import find_and_run_script
from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.constants import DEFAULT_CONFIG_PATH
from syftbox.lib.exceptions import ClientConfigException
from syftbox.lib.workspace import SyftWorkspace

app = Typer(
    name="SyftBox Apps",
    help="Manage SyftBox apps",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

CONFIG_OPTS = Option("-c", "--config", "--config_path", help="Path to the SyftBox config")
CALLED_BY_OPTS = Option("--called-by", help="The context from which the command is called", hidden=True)
REPO_ARGS = Argument(..., show_default=False, help="SyftBox App git repo URL")
BRANCH_OPTS = Option("-b", "--branch", help="git branch name")
UNINSTALL_ARGS = Argument(..., show_default=False, help="Name of the SyftBox App to uninstall")
APP_ENV_SCRIPT = """
if [ ! -d .venv ]; then
    uv venv .venv
fi
. .venv/bin/activate
"""


@app.command()
def list(config_path: Annotated[Path, CONFIG_OPTS] = DEFAULT_CONFIG_PATH) -> None:
    """List all installed Syftbox apps"""
    workspace = get_workspace(config_path)
    result = list_app(workspace)

    if len(result.apps) == 0:
        rprint(f"No apps installed in '{result.apps_dir}'")
        sys.exit(0)

    rprint(f"Apps installed in '{result.apps_dir}'")
    for app in result.apps:
        rprint(f"- [bold cyan]{app.name}[/bold cyan]")


@app.command()
def install(
    repository: Annotated[str, REPO_ARGS],
    branch: Annotated[str, BRANCH_OPTS] = "main",
    config_path: Annotated[Path, CONFIG_OPTS] = DEFAULT_CONFIG_PATH,
    called_by: Annotated[str, CALLED_BY_OPTS] = "user",
) -> None:
    """Install a new Syftbox app"""
    context = get_syftbox_context(config_path)
    result = install_app(context.workspace, repository, branch)
    if result.error:
        rprint(f"[bold red]Error:[/bold red] {result.error}")
        raise Exit(1)

    try:
        context.client.log_analytics_event(
            event_name="app_install",
            app_name=result.app_name,
            called_by=called_by,
        )
    except Exception as e:
        logger.debug(f"Failed to log analytics event: {e}")

    rprint(f"Installed app [bold]'{result.app_name}'[/bold]\nLocation: '{result.app_path}'")


@app.command()
def uninstall(
    app_name: Annotated[str, UNINSTALL_ARGS],
    config_path: Annotated[Path, CONFIG_OPTS] = DEFAULT_CONFIG_PATH,
) -> None:
    """Uninstall a Syftbox app"""
    workspace = get_workspace(config_path)
    result = uninstall_app(app_name, workspace)
    if not result:
        rprint(f"[bold red]Error:[/bold red] '{app_name}' app not found")
        raise Exit(1)

    rprint(f"Uninstalled app [bold]'{app_name}'[/bold] from '{result}'")


@app.command()
def run(
    app_name: str,
    config_path: Annotated[Path, CONFIG_OPTS] = DEFAULT_CONFIG_PATH,
) -> None:
    """Run a Syftbox app"""
    workspace = get_workspace(config_path)

    extra_args: List[str] = []
    try:
        rprint(f"Running [bold]'{app_name}'[/bold]\nLocation: '{workspace.apps}'\n")
        result = find_and_run_script(workspace.apps / app_name, extra_args, config_path)
        rprint("[bold yellow]stdout:[/bold yellow]")
        print(result.stdout)
        rprint("[bold yellow]stderr:[/bold yellow]")
        print(result.stderr)
    except Exception as e:
        rprint("[bold red]Error:[/bold red]", e)
        raise Exit(1)


@app.command(rich_help_panel="General Options")
def env(with_syftbox: bool = False) -> None:
    """Setup virtual env for app. With option to install syftbox matching client version"""

    script = APP_ENV_SCRIPT
    if with_syftbox:
        script += f"\nuv pip install -U syftbox=={__version__}"
    print(script)


# @app.command()
# def update(
#     app_name: Annotated[str, Argument(help="Name of the app to uninstall")],
#     config_path: Annotated[Path, CONFIG_OPTS] = DEFAULT_CONFIG_PATH,
# ):
#     """Update a Syftbox app"""
#     pass


def get_syftbox_context(config_path: Path) -> SyftBoxContextInterface:
    try:
        conf = SyftClientConfig.load(config_path)
        context = SyftBoxRunner(conf).context
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


def get_workspace(config_path: Path) -> SyftWorkspace:
    try:
        conf = SyftClientConfig.load(config_path)
        return SyftWorkspace(conf.data_dir)
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
