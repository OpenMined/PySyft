# stdlib
from typing import Tuple


def is_broadcastable(shape1: Tuple[int], shape2: Tuple[int]) -> bool:
    """Helper function to determine if Tensor Operations can be broadcast
    inputs:
    shape1, shape 2: shapes of numpy arrays/syft tensors

    outputs:
    True or False
    """
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True
