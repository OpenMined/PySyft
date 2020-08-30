from benchmark_sigmoid import benchmark_sigmoid
from workers_initialization import workers, hook


def test_sigmoid_chebyshev_1(benchmark):
    worker = workers(hook())
    benchmark(benchmark_sigmoid, "chebyshev", 1, worker)
