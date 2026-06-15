from typing import Any

from syft_notebook_ui.types import TableDict, TableList


def display(obj: Any) -> Any:
    """Convert an object to a displayable object, if possible."""

    if isinstance(obj, list):
        return TableList(obj)
    elif isinstance(obj, dict):
        return TableDict(obj)

    return obj
