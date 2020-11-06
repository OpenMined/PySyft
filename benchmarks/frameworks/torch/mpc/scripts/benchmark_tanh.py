from abstract.approx_benchmark import ApproxBenchmark
from benchmark_sample_data import benchmark_data_tanh


class TanhBenchmark(ApproxBenchmark):
    @staticmethod
    def _target_operation(tensor, method=None):
        if method is None:
            return tensor.tanh()
        return tensor.tanh(method=method)


SAVE_PATH = "../graphs/tanh_function_approximations_benchmark.png"
TanhBenchmark.run(benchmark_data_tanh, save_path=SAVE_PATH)
