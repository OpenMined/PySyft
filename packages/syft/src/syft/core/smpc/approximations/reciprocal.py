"""function used to calculate reciprocal of a given tensor."""

# future
from __future__ import annotations

# stdlib
from typing import TYPE_CHECKING

# relative
from .exp import exp
from .log import log
from .utils import modulus
from .utils import sign

if TYPE_CHECKING:
    # relative
    from ...tensor.smpc.mpc_tensor import MPCTensor


def reciprocal(data: MPCTensor, method: str = "NR", nr_iters: int = 5) -> MPCTensor:
    """Calculate the reciprocal using the algorithm specified in the method args.
    Ref: https://github.com/facebookresearch/CrypTen

    Args:
        data: input data
        nr_iters: Number of iterations for Newton-Raphson
        method: 'NR' - `Newton-Raphson`_ method computes the reciprocal using iterations
                of :math:`x_{i+1} = (2x_i - data * x_i^2)` and uses
                :math:`3*exp(-(x-.5)) + 0.003` as an initial guess by default
                'log' -  Computes the reciprocal of the input from the observation that:
                        :math:`x^{-1} = exp(-log(x))`

    Returns:
        Reciprocal of `data`

    Raises:
        ValueError: if the given method is not supported
    """
    method = method.lower()

    if method == "nr":
        new_data = modulus(data)
        result = exp(new_data * -1 + 0.5) * 3 + 0.003
        for i in range(nr_iters):
            result = result * 2 - result * result * new_data
        return result * sign(data)
    elif method == "log":
        new_data = modulus(data)
        return exp(-1 * log(new_data)) * sign(data)
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")
