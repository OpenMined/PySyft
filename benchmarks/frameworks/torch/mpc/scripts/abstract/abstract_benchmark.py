from abc import ABC
from abc import abstractclassmethod

from benchmarks.frameworks.torch.mpc.scripts.abstract.workers_initialization import workers, hook


class AbstractBenchmark(ABC):
    @abstractclassmethod
    def _benchmark(cls, method, prec_frac, workers, **kwargs):
        """Operation to benchmark
        NOTE: This function must be implemented.

        Args:
            method (str): the name of the method / protocol
            prec_frac (int): precision_value
            workers (dict): workers used for sharing data
        """
        pass

    @abstractclassmethod
    def _plot(cls, benchmark_data, worker, save_path):
        """Benchmark specific operations and store the results as images.
        NOTE: This function must be implemented.

        Args:
            benchmark_data (list): the sample data
            worker (dict): workers used for sharing data
            save_path (str): path where result images will be stored.
        """
        pass

    @classmethod
    def run(cls, benchmark_data, save_path):
        """Benchmark specific operations and store the results as images.

        Args:
            benchmark_data (list): the sample data
            save_path (str): path where result images will be stored.
        """
        worker = workers(hook())
        cls._plot(benchmark_data, worker, save_path)
