# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def chebyshev_series(func, width, terms):
    r"""
    Computes Chebyshev coefficients
    For n = terms, the ith Chebyshev series coefficient is
    .. math::
        c_i = 2/n \sum_{k=1}^n \cos(j(2k-1)\pi / 4n) f(w\cos((2k-1)\pi / 4n))
    Args:
        func (function): function to be approximated
        width (int): approximation will support inputs in range [-width, width]
        terms (int): number of Chebyshev terms used in approximation
    Returns:
        Chebyshev coefficients with shape equal to num of terms.
    """
    n_range = torch.arange(start=0, end=terms).float()
    x = width * torch.cos((n_range + 0.5) * np.pi / terms)
    y = func(x)
    cos_term = torch.cos(torch.ger(n_range, n_range + 0.5) * np.pi / terms)
    coeffs = (2 / terms) * torch.sum(y * cos_term, axis=1)
    return coeffs


def chebyshev_polynomials(tensor, terms=32):
    r"""
    Evaluates odd degree Chebyshev polynomials at x
    Chebyshev Polynomials of the first kind are defined as
    .. math::
        P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)
    Args:
        tensor (torch.tensor): input at which polynomials are evaluated
        terms (int): highest degree of Chebyshev polynomials.
                     Must be even and at least 6.
    """
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    polynomials = [tensor.clone()]
    y = 4 * tensor ** 2 - 2
    z = y - 1
    polynomials.append(z.mul(tensor))

    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(next_polynomial)

    return torch.stack(polynomials)
