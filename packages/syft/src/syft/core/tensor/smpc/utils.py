"""Utils functions that might be used into any module."""

# stdlib
from typing import Any

# third party
import numpy as np
import torch


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


def is_int_tensor(maybe_tensor: Any) -> bool:
    """Check if an object is an int type tensor

    Args:
        maybe_tensor (Any): Object to be checked

    Returns:
        bool: True if the object is a tensor of type integer or False if not
    """
    return torch.is_tensor(maybe_tensor) and not (
        torch.is_floating_point(maybe_tensor) and torch.is_complex(maybe_tensor)
    )


def is_int_array(maybe_array: Any) -> bool:
    """Check if an object is an int type numpy array

    Args:
        maybe_array (Any): Object to be checked

    Returns:
        bool: True if the object is an array of type integer or False if not
    """
    return isinstance(maybe_array, np.ndarray) and np.issubsctype(
        maybe_array, np.integer
    )
