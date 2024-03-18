# stdlib
from collections.abc import Callable
import importlib
from typing import Any

# third party
from typer import Typer


def add_subcmd(app: Typer, commands: list[Callable]) -> None:
    for cmd in commands:
        app.command()(cmd)


def add_cmd(app: Typer, mod_name: str) -> None:
    # TODO: pyinstaller is very unhappy with dynamic imports
    # to fix this we need to make the pyinstaller spec dynamically pick up all modules under 'src'
    # till then we continue to use typer.add_typer(command.cmd, name="command")
    module: Any = importlib.import_module(f"src.{mod_name}")
    app.add_typer(module.app, name=mod_name)
