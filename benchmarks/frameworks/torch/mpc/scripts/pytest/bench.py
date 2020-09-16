"""
Define benchmark tests
"""

from workers_initialization import workers, hook
from benchmark_functions import sigmoid

# Initialize workers globally for the tests
worker = workers(hook())


def test_sigmoid_chebyshev_1(benchmark):
    """
    Test sigmoid aproximation with chebyshev method an
    precision value of 1
    """
    benchmark(sigmoid, "chebyshev", 1, worker)
