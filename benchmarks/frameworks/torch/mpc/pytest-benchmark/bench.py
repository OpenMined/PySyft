"""
Define benchmark tests
"""

from workers_initialization import workers, hook
from benchmark_functions import sigmoid, tanh

# Initialize workers globally for the tests
worker = workers(hook())


def test_sigmoid_chebyshev(benchmark):
    """
    Test sigmoid aproximation with chebyshev method and
    precision value of 1
    """
    benchmark(sigmoid, "chebyshev", 1, worker)


def test_tanh_chebyshev(benchmark):
    """
    Test tanh aproximation with chebyshev method and
    precision value of 1
    """
    benchmark(tanh, "chebyshev", 1, worker)
