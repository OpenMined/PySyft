# third party
from typer import Typer

# relative
from ..core.register import add_subcmd
from .create import create

__all__ = ["cmd"]

cmd = Typer(no_args_is_help=True)

add_subcmd(cmd, [create])
