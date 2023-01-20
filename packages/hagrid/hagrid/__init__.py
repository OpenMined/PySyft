from .util import verify_git_installation  # noqa

# stdlib
import sys
from typing import Any

# relative
from .cli import check_status as check  # noqa: F401
from .quickstart_ui import QuickstartUI
from .version import __version__  # noqa: F401
from .wizard_ui import WizardUI


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
