"""
Define benchmark tests
"""
from benchmarks.frameworks.torch.mpc.pytestbenchmark.benchmark_functions import sigmoid, tanh


def test_sigmoid_chebyshev(benchmark, workers):
    """
    Test sigmoid approximation with chebyshev method and
    precision value of 4
    """
    benchmark(sigmoid, "chebyshev", 4, workers)


def test_sigmoid_maclaurin(benchmark, workers):
    """
    Test sigmoid approximation with maclaurin method and
    precision value of 4
    """
    benchmark(sigmoid, "maclaurin", 4, workers)


def test_sigmoid_exp(benchmark, workers):
    """
    Test sigmoid approximation with exp method and
    precision value of 4
    """
    benchmark(sigmoid, "exp", 4, workers)


def test_tanh_chebyshev(benchmark, workers):
    """
    Test tanh approximation with chebyshev method and
    precision value of 4
    """
    benchmark(tanh, "chebyshev", 4, workers)


def test_tanh_sigmoid(benchmark, workers):
    """
    Test tanh approximation with sigmoid method and
    precision value of 4
    """
    benchmark(tanh, "sigmoid", 4, workers)
