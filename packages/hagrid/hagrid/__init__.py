# stdlib
import sys
from typing import Any

# relative
from .quickstart import Quickstart
from .wizard import Wizard


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
def _quickstart() -> Quickstart:
    return Quickstart()


@module_property
def _wizard() -> Wizard:
    return Wizard()
