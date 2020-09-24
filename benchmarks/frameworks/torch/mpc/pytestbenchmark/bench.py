"""
Define benchmark tests
"""
from benchmarks.frameworks.torch.mpc.pytestbenchmark.benchmark_functions import sigmoid, tanh


def test_sigmoid_chebyshev(benchmark, workers):
    """
    Test sigmoid approximation with chebyshev method and
    precision value of 1
    """
    benchmark(sigmoid, "chebyshev", 4, workers)


def test_tanh_chebyshev(benchmark, workers):
    """
    Test tanh approximation with chebyshev method and
    precision value of 1
    """
    benchmark(tanh, "chebyshev", 4, workers)
