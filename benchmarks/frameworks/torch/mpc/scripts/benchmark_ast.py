import torch
import timeit
import matplotlib.pyplot as plt

from benchmarks.frameworks.torch.mpc.scripts.workers_initialization import workers, hook
from benchmarks.frameworks.torch.mpc.scripts.benchmark_sample_data import b_data_share_get


def benchmark_share_get(workers, protocol, dtype, n_workers):
    """
    This function approximates the sigmoid function using a given method.

    Args:
        workers (dict): workers used for sharing data
        protocol (str): the name of the protocol
        dtype (int): data type
        n_workers (int): number of workers

    """
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )

    share_holders = [alice, bob, charlie]
    kwargs = dict(protocol=protocol, crypto_provider=james, dtype=dtype)

    t = torch.tensor([1, 2, 3])

    x = t.share(*share_holders[:n_workers], **kwargs)
    x = x.get()


def benchmark_share_get_plot(b_data_share_get):
    """
        This function plots the graph for various protocols benchmarks for additive
        shared tensors

        Args:
            b_data_share_get (list): the sample data to approximate

        Returns:
            benchmark_share_get.png (png): plotted graph in graph/ast_benchmarks directory
        """
    # initializing workers
    worker = workers(hook())

    # available protocols
    protocols = ["snn", "fss"]

    # initializing graph plot
    fig, ax = plt.subplots()

    for protocol in protocols:

        # list for handling graph data
        x_data = []
        y_data = []

        for data in b_data_share_get:

            # getting value from b_data_share_get
            dtype, n_workers = data

            # temporary list for calculating average execution time and error
            temp_time_taken = []

            for i in range(10):
                start_time = timeit.default_timer()
                benchmark_share_get(worker, protocol, dtype, n_workers)
                time_taken = timeit.default_timer() - start_time
                temp_time_taken.append(time_taken)

            final_time_taken = sum(temp_time_taken) / len(temp_time_taken)
            final_time_taken *= 1000
            x_data.append(dtype + str(" / ") + str(n_workers))
            y_data.append(final_time_taken)

        ax.plot(x_data, y_data, label=protocol, linestyle="-")
        x_data.clear()
        y_data.clear()

    ax.set_xlabel("dtype / n_workers")
    ax.set_ylabel("Execution Time (ms)")
    ax.legend(bbox_to_anchor=(1, 1.22), loc="upper right", title="Protocols", fontsize="small")
    plt.tight_layout()
    plt.savefig("../graphs/ast_benchmarks/benchmark_share_get.png")
    # plt.show()


# calling benchmark_share_get_plot function
benchmark_share_get_plot(b_data_share_get)
