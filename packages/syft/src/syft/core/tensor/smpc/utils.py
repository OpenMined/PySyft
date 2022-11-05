"""Utils functions that might be used into any module."""

# stdlib
from functools import lru_cache
import operator
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import cast

# third party
import numpy as np

# relative
from ..config import DEFAULT_INT_NUMPY_TYPE

RING_SIZE_TO_TYPE: Dict[int, np.dtype] = {
    2**32: np.dtype("int32"),
    2**64: np.dtype("int64"),
    2: np.dtype("bool"),  # Special case: need to do reconstruct and share with XOR
}

TYPE_TO_RING_SIZE: Dict[np.dtype, Optional[int]] = {
    v: k for k, v in RING_SIZE_TO_TYPE.items()
}


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


NUMPY_OPS = {"concatenate"}
OPERATOR_OPS = {
    "add",
    "sub",
    "div",
    "truediv",
    "mul",
    "eq",
    "ne",
    "gt",
    "lt",
    "ge",
    "le",
    "matmul",
    "pos",
    "pow",
    "lshift",
    "rshift",
    "xor",
}


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
    dummy_x = np.empty(x_shape, dtype=DEFAULT_INT_NUMPY_TYPE)
    dummy_y = np.empty(y_shape, dtype=DEFAULT_INT_NUMPY_TYPE)
    if op_str in OPERATOR_OPS:
        op = getattr(operator, op_str)
        res = op(dummy_x, dummy_y).shape
    elif op_str in NUMPY_OPS:
        res = getattr(np, op_str)([dummy_x, dummy_y]).shape
    else:
        res = getattr(dummy_x, op_str)(dummy_y).shape

    res = cast(Tuple[int], res)
    return res


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
            f"Expected the same ring size for x and y ({x_ring_size} vs {y_ring_size})"
        )

    return x_ring_size


# Code Adapted from Crypten Project:  https://github.com/facebookresearch/CrypTen


def count_wraps(share_list: List[np.ndarray]) -> np.ndarray:
    """Count overflows and underflows if we reconstruct the original value.

    The code is adapted from the Crypten Project.

    Args:
        share_list (List[ShareTensor]): List of the shares.

    Returns:
        torch.Tensor: The number of wraparounds.
    """
    res = np.zeros_like(share_list[0], dtype=DEFAULT_INT_NUMPY_TYPE)
    prev_share = share_list[0]
    for cur_share in share_list[1:]:
        next_share = cur_share + prev_share

        # If prev and current shares are negative,
        # but the result is positive then is an underflow
        res -= (prev_share < 0) & (cur_share < 0) & (next_share > 0)

        # If prev and current shares are positive,
        # but the result is positive then is an overflow
        res += (prev_share > 0) & (cur_share > 0) & (next_share < 0)
        prev_share = next_share

    return res
