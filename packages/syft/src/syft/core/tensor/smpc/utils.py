"""Utils functions that might be used into any module."""

# stdlib
from functools import lru_cache
from typing import Any
from typing import Dict

# third party
import numpy as np


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


RING_SIZE_TO_TYPE: Dict[int, np.dtype] = {
    2 ** 32: np.int32,
    2: np.bool_,  # Special case: need to do reconstruct and share with XOR
}


@lru_cache()
def get_nr_bits(ring_size: int) -> int:
    """Get number of bits.

    Args:
        ring_size (int): Ring Size.

    Returns:
        int: Bit length.
    """
    return (ring_size - 1).bit_length()
