from abstract.approx_benchmark import ApproxBenchmark
from benchmark_sample_data import benchmark_data_sigmoid


class SigmoidBenchmark(ApproxBenchmark):
    @staticmethod
    def _target_operation(tensor, method=None):
        if method is None:
            return tensor.sigmoid()
        return tensor.sigmoid(method=method)


SAVE_PATH = "../graphs/sigmoid_function_approximations_benchmark.png"
SigmoidBenchmark.run(benchmark_data_sigmoid, save_path=SAVE_PATH)
