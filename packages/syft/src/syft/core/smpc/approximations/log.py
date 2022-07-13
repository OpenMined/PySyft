"""fucntion used to calculate log of given tensor."""
# future
from __future__ import annotations

# stdlib
from typing import TYPE_CHECKING
from typing import Union

# relative
from .exp import exp

if TYPE_CHECKING:
    # relative
    from ...tensor.smpc.mpc_tensor import MPCTensor


def log(
    data: MPCTensor, iterations: int = 2, exp_iterations: int = 8
) -> Union[float, MPCTensor]:
    """Approximates the natural logarithm using 8th order modified Householder iterations.
        Recall that Householder method is an algorithm to solve a non linear equation f(x) = 0.
        Here  f: x -> 1 - C * exp(-x)  with C = data
        Iterations are computed by:
            y_0 = some constant
            h = 1 - data * exp(-y_n)
            y_{n+1} = y_n - h * (1 + h / 2 + h^2 / 3 + h^3 / 6 + h^4 / 5 + h^5 / 7)

    Args:
        data: tensor whose log has to be calculated
        iterations (int): number of iterations for 6th order modified
            Householder approximation.
        exp_iterations (int): number of iterations for limit approximation of exp

    Returns:
        MPCTensor: Calculated log value of given tensor
    Ref: https://github.com/LaRiffle/approximate-models
    """
    y = data / 31 + 1.59 - 20 * exp((-2 * data - 1.4), iterations=exp_iterations)

    # 6th order Householder iterations
    for i in range(iterations):
        h = [1 - data * exp((-1 * y), iterations=exp_iterations)]
        for j in range(1, 5):
            h.append(h[-1] * h[0])

        y -= h[0] * (1 + h[0] / 2 + h[1] / 3 + h[2] / 4 + h[3] / 5 + h[4] / 6)

    return y
