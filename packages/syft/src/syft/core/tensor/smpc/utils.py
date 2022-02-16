"""Utils functions that might be used into any module."""

# stdlib
from functools import lru_cache
import operator
from typing import Any
from typing import Dict
from typing import Tuple
from typing import cast

# third party
import numpy as np

RING_SIZE_TO_TYPE: Dict[int, np.dtype] = {
    2**32: np.dtype("int32"),
    2: np.dtype("bool"),  # Special case: need to do reconstruct and share with XOR
}

TYPE_TO_RING_SIZE: Dict[np.dtype, int] = {v: k for k, v in RING_SIZE_TO_TYPE.items()}


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


@lru_cache()
def get_nr_bits(ring_size: int) -> int:
    """Get number of bits.

    Args:
        ring_size (int): Ring Size.

    Returns:
        int: Bit length.
    """
    return (ring_size - 1).bit_length()


@lru_cache(maxsize=128)
def get_shape(
    op_str: str,
    x_shape: Tuple[int],
    y_shape: Tuple[int],
) -> Tuple[int]:
    """Get the shape of apply an operation on two values

    Args:
        op_str (str): the operation to be applied
        x_shape (Tuple[int]): the shape of op1
        y_shape (Tuple[int]): the shape of op2

    Returns:
        The shape of the result
    """
    op = getattr(operator, op_str)
    res = op(np.empty(x_shape), np.empty(y_shape)).shape
    res = cast(Tuple[int], res)
    return tuple(res)  # type: ignore


@lru_cache(maxsize=128)
def get_ring_size(
    x_ring_size: int,
    y_ring_size: int,
) -> int:
    """Get the ring_size of apply an operation on two values

    Args:
        x_ring_size (int): the ring size of op1
        y_ring_size (int): the ring size of op2

    Returns:
        The ring size of the result
    """
    if x_ring_size != y_ring_size:
        raise ValueError(
            "Expected the same ring size for x and y ({x_ring_size} vs {y_ring_size})"
        )

    return x_ring_size
