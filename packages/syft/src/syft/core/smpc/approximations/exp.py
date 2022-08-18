"""Function used to calculate exp of a given tensor."""
# future
from __future__ import annotations

# stdlib
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    # relative
    from ...tensor import Tensor
    from ...tensor.smpc.mpc_tensor import MPCTensor


def exp(
    value: Union[Tensor, MPCTensor, int, float], iterations: int = 8
) -> Union[MPCTensor, float, Tensor]:
    """Approximates the exponential function using a limit approximation.

    exp(x) = lim_{n -> infty} (1 + x / n) ^ n
    Here we compute exp by choosing n = 2 ** d for some large d equal to
    iterations. We then compute (1 + x / n) once and square `d` times.

    Args:
        value: tensor whose exp is to be calculated
        iterations (int): number of iterations for limit approximation
    Ref: https://github.com/LaRiffle/approximate-models

    Returns:
        MPCTensor: the calculated exponential of the given tensor
    """
    result = (value / 2**iterations) + 1
    for _ in range(iterations):
        result = result * result
    return result
