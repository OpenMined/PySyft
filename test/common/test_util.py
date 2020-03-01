import torch
import itertools

from syft.common.util import chebyshev_series, chebyshev_polynomials


def test_chebyshev_polynomials():
    """Tests evaluation of chebyshev polynomials"""
    sizes = [(1, 10), (3, 5), (3, 5, 10)]
    possible_terms = [6, 40]
    tolerance = 0.05

    for size, terms in itertools.product(sizes, possible_terms):
        tensor = torch.rand(torch.Size(size)) * 42 - 42
        result = chebyshev_polynomials(tensor, terms)

        # check number of polynomials
        assert result.shape[0] == terms // 2

        assert torch.all(result[0] == tensor), "first term is incorrect"

        second_term = 4 * tensor ** 3 - 3 * tensor
        diff = (result[1] - second_term).abs()
        norm_diff = diff.div(result[1].abs() + second_term.abs())
        assert torch.all(norm_diff <= tolerance), "second term is incorrect"


def test_chebyshev_series():
    """Checks coefficients returned by chebyshev_series are correct"""
    for width, terms in [(6, 10), (6, 20)]:
        result = chebyshev_series(torch.tanh, width, terms)

        # check shape
        assert result.shape == torch.Size([terms])

        # check terms
        assert result[0] < 1e-4
        assert torch.isclose(result[-1], torch.tensor(3.5e-2), atol=1e-1)
