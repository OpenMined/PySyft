"""Utils functions that might be used into any module."""

# stdlib
from typing import Any


def ispointer(obj: Any) -> bool:
    """Check if a given obj is a pointer (is a remote object).

    Args:
        obj (Any): Object.

    Returns:
        bool: True (if pointer) or False (if not).
    """
    if type(obj).__name__.endswith("Pointer") and hasattr(obj, "id_at_location"):
        return True
    return False
