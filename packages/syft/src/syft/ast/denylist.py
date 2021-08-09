"""This module contains denylist.

`denylist` contains list of paths that are explicitly denied and shouldn't be added to ast for security issues."""

# stdlib
import re
from typing import Set

_denylist: Set[str] = {
    "pandas.read_",
}

_denylist_re = list(map(re.compile, _denylist))  # type: ignore


def deny(path: str) -> bool:
    """Check if path is in denylist.

    Args:
        path: the path of the `Attribute` to be added to AST.

    Returns:
        bool: True if the path is denied. (Don't add this path)
    """
    return any(bool(regex.match(path)) for regex in _denylist_re)  # type: ignore
