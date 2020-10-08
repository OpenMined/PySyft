import abc
import timeit

import torch
import matplotlib.pyplot as plt

from abstract.abstract_benchmark import AbstractBenchmark


class ApproxBenchmark(AbstractBenchmark, metaclass=abc.ABCMeta):
    SAMPLE_TENSOR = torch.tensor([1.23212])

    @abc.abstractstaticmethod
    def _target_operation(tensor, method=None):
        pass

    @classmethod
    def _benchmark(cls, method, prec_frac, workers, **kwrags):
        alice = workers["alice"]
        bob = workers["bob"]
        james = workers["james"]

        t = cls.SAMPLE_TENSOR
        t_sh = t.fix_precision(precision_fractional=prec_frac).share(
            alice, bob, crypto_provider=james
        )
        r_sh = cls._target_operation(t_sh, method=method)
        r = r_sh.get().float_prec()
        t = cls._target_operation(t)

        diff = (r - t).abs().max()
        return diff.item()

    @classmethod
    def _plot(cls, benchmark_data, worker, save_path):
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        for method, prec_frac in benchmark_data:
            x_data = []
            y1_data = []
            y2_data = []

            for prec_value in range(1, prec_frac + 1):
                temp_time_taken = []
                temp_error = []

                for _ in range(10):
                    start_time = timeit.default_timer()
                    error = cls._benchmark(method, prec_frac, worker)
                    time_taken = timeit.default_timer() - start_time
                    temp_time_taken.append(time_taken)
                    temp_error.append(error)

                final_time_taken = sum(temp_time_taken) / len(temp_time_taken)
                final_time_taken *= 1000
                final_error = sum(temp_error) / len(temp_error)

                x_data.append(prec_value)
                y1_data.append(final_time_taken)
                y2_data.append(final_error)

            ax1.plot(x_data, y1_data, label=method, linestyle="-")
            ax2.plot(x_data, y2_data, label=method, linestyle="--")

        ax1.set_xlabel("Precision Value")
        ax1.set_ylabel("Execution Time (ms)")
        ax2.set_ylabel("Error")
        ax1.legend(bbox_to_anchor=(1, 1.3), loc="upper right", title="Method", fontsize="small")
        ax2.legend(bbox_to_anchor=(0, 1.3), loc="upper left", title="Error", fontsize="small")
        plt.tight_layout()
        plt.savefig(save_path)
