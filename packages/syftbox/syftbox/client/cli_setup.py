"""
SyftBox CLI - Setup scripts
"""

import json
import shutil
from pathlib import Path

import httpx
import typer
from rich import print as rprint
from rich.prompt import Confirm, Prompt
from typing_extensions import Optional

from syftbox import __version__
from syftbox.client.auth import authenticate_user
from syftbox.client.core import METADATA_FILENAME
from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.constants import DEFAULT_DATA_DIR
from syftbox.lib.exceptions import ClientConfigException
from syftbox.lib.http import HEADER_SYFTBOX_USER, SYFTBOX_HEADERS
from syftbox.lib.validators import DIR_NOT_EMPTY, is_valid_dir, is_valid_email
from syftbox.lib.workspace import SyftWorkspace

__all__ = ["setup_config_interactive"]


def is_empty(data_dir: Path) -> bool:
    """True if the data_dir is empty"""
    return not any(data_dir.iterdir())


def has_old_syftbox_version(data_dir: Path) -> bool:
    """True if the data_dir was created with an older version of SyftBox"""
    metadata_file = data_dir / METADATA_FILENAME
    if not metadata_file.exists():
        return True
    metadata = json.loads(metadata_file.read_text())
    current_version = __version__
    old_version = metadata.get("version", None)
    return old_version != current_version


def prompt_delete_old_data_dir(data_dir: Path) -> bool:
    msg = f"[yellow]Found old SyftBox folder at {data_dir}.[/yellow]\n"
    msg += "[yellow]Press Y to remove the old folder and download it from the server [bold](recommended)[/bold]. Press N to keep the old folder and migrate it.[/yellow]"
    return Confirm.ask(msg)


def get_migration_decision(data_dir: Path) -> bool:
    migrate_datasite = False
    if data_dir.exists():
        if is_empty(data_dir):
            migrate_datasite = False
        elif has_old_syftbox_version(data_dir):
            # we need this extra if because we do 2 things:
            # 1. determine if we want to remove
            # 2. determine if we want to migrate
            if prompt_delete_old_data_dir(data_dir):
                rprint("Removing old syftbox folder")
                apps_dir = SyftWorkspace(data_dir).apps
                paths_to_exclude = [apps_dir]
                # Remove everything except the paths in paths_to_exclude
                for item in data_dir.iterdir():
                    if item not in paths_to_exclude:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                migrate_datasite = False
            else:
                migrate_datasite = True
    return migrate_datasite


def setup_config_interactive(
    config_path: Path,
    email: str,
    data_dir: Path,
    server: str,
    port: int,
    skip_auth: bool = False,
    skip_verify_install: bool = False,
) -> SyftClientConfig:
    """Setup the client configuration interactively. Called from CLI"""

    config_path = config_path.expanduser().resolve()
    conf: Optional[SyftClientConfig] = None
    if data_dir:
        data_dir = data_dir.expanduser().resolve()

    # try to load the existing config
    try:
        conf = SyftClientConfig.load(config_path)
    except ClientConfigException:
        pass

    if not conf:
        # first time setup
        if not data_dir or data_dir == DEFAULT_DATA_DIR:
            data_dir = prompt_data_dir()

        if not email:
            email = prompt_email()

        # create a new config with the input params
        conf = SyftClientConfig(
            path=config_path,
            sync_folder=data_dir,
            email=email,
            server_url=server,
            port=port,
        )
    else:
        if server and server != conf.server_url:
            conf.set_server_url(server)
        if port != conf.client_url.port:
            conf.set_port(port)

    # Short-lived client for all pre-authentication requests
    login_client = httpx.Client(
        base_url=str(conf.server_url),
        headers={
            **SYFTBOX_HEADERS,
            HEADER_SYFTBOX_USER: conf.email,
        },
        transport=httpx.HTTPTransport(retries=10),
    )
    if not skip_verify_install:
        verify_installation(conf, login_client)

    if not skip_auth:
        conf.access_token = authenticate_user(conf, login_client)

    # DO NOT SAVE THE CONFIG HERE.
    # We don't know if the client will accept the config yet
    return conf


def prompt_data_dir(default_dir: Path = DEFAULT_DATA_DIR) -> Path:
    prompt_dir = "[bold]Where do you want SyftBox to store data?[/bold] [grey70]Press Enter for default[/grey70]"
    prompt_overwrite = "[bold yellow]Directory '{sync_folder}' is not empty![/bold yellow] Do you want to overwrite it?"

    while True:
        sync_folder = Prompt.ask(prompt_dir, default=str(default_dir))
        valid, reason = is_valid_dir(sync_folder)
        if reason == DIR_NOT_EMPTY:
            overwrite = Confirm.ask(prompt_overwrite.format(sync_folder=sync_folder))
            if not overwrite:
                continue
            valid = True

        if not valid:
            rprint(f"[bold red]{reason}[/bold red] '{sync_folder}'")
            continue

        path = Path(sync_folder).expanduser().resolve()
        rprint(f"Selected directory [bold]'{path}'[/bold]")
        return path


def prompt_email() -> str:
    while True:
        email = Prompt.ask("[bold]Enter your email address[/bold]")
        if not is_valid_email(email):
            rprint(f"[bold red]Invalid email[/bold red]: '{email}'")
            continue
        return email


def verify_installation(conf: SyftClientConfig, client: httpx.Client) -> None:
    try:
        response = client.get("/info?verify_installation=1")

        response.raise_for_status()

    except (httpx.HTTPError, KeyError):
        should_continue = Confirm.ask(
            "\n[bold red]Could not connect to the SyftBox server, continue anyway?[/bold red]"
        )
        if not should_continue:
            raise typer.Exit()
