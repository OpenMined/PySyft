from .git_check import verify_git_installation  # noqa

# stdlib
import sys
from typing import Any

# third party
import rich
from rich.text import Text

# relative
from .cli import check_status as check  # noqa: F401
from .quickstart_ui import QuickstartUI
from .version import __version__  # noqa: F401
from .wizard_ui import WizardUI

from .orchestra import Orchestra  # noqa

console = rich.get_console()
table = rich.table.Table(show_header=False)
table.add_column(justify="center", overflow="fold")
table.add_row(
    "🚨🚨🚨 Hagrid has been deprecated. 🚨🚨🚨",
    style=rich.style.Style(
        bold=True,
        color="red",
    ),
)
link = "https://github.com/OpenMined/PySyft/tree/dev/notebooks/tutorials/deployments"
link_text = Text(link, style="link " + link + " cyan")
normal_text = Text("Please refer to ")
normal_text.append(link_text)
normal_text.append(" for the new deployment instructions.")
table.add_row(normal_text)
console.print(table)


def module_property(func: Any) -> None:
    """Decorator to turn module functions into properties.
    Function names must be prefixed with an underscore."""
    module = sys.modules[func.__module__]

    def base_getattr(name: str) -> None:
        raise AttributeError(f"module '{module.__name__}' has no attribute '{name}'")

    old_getattr = getattr(module, "__getattr__", base_getattr)

    def new_getattr(name: str) -> Any:
        if f"_{name}" == func.__name__:
            return func()
        else:
            return old_getattr(name)

    module.__getattr__ = new_getattr  # type: ignore
    return func


@module_property
def _quickstart() -> QuickstartUI:
    return QuickstartUI()


@module_property
def _wizard() -> WizardUI:
    return WizardUI()
