import torch
import timeit
import matplotlib.pyplot as plt

from benchmarks.frameworks.torch.mpc.scripts.workers_initialization import workers, hook
from benchmarks.frameworks.torch.mpc.scripts.benchmark_sample_data import benchmark_data_sigmoid


def benchmark_sigmoid(method, prec_frac, workers):
    """
    This function approximates the sigmoid function using a given method.

    Args:
        method (str): the name of the method for approximation
        prec_frac (int): precision value
        workers (dict): workers used for sharing data

    Returns:
        diff (int): the difference between the syft and torch approximated value

    """
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    t = torch.tensor([1.23212])
    t_sh = t.fix_precision(precision_fractional=prec_frac).share(alice, bob, crypto_provider=james)
    r_sh = t_sh.sigmoid(method=method)
    r = r_sh.get().float_prec()
    t = t.sigmoid()
    # Calculation of the difference between FPT and normal sigmoid (error)
    diff = (r - t).abs().max()
    return diff.item()


def sigmoid_approximation_plot(benchmark_data_sigmoid):
    """
    This function plots the graph for various sigmoidal approximation benchmarks namely
    'chebyshev', 'maclaurin', 'exp'.

    Args:
        benchmark_data_sigmoid (list): the sample data to approximate

    Returns:
        sigmoid_function_approximations_benchmark (png): plotted graph in graph directory
    """

    # initializing workers
    worker = workers(hook())

    # initializing graph plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # list for handling graph data
    x_data = []
    y_data = []
    y2_data = []

    for data in benchmark_data_sigmoid:

        # getting value from benchmark_data_sigmoid
        method, prec_frac = data

        for precision_value in range(1, (prec_frac + 1)):

            # temporary list for calculating average execution time and error
            temp_time_taken = []
            temp_error = []

            for i in range(10):
                start_time = timeit.default_timer()
                error = benchmark_sigmoid(method, precision_value, worker)
                time_taken = timeit.default_timer() - start_time
                temp_time_taken.append(time_taken)
                temp_error.append(error)

            final_time_taken = sum(temp_time_taken) / len(temp_time_taken)
            final_time_taken *= 1000
            final_error = sum(temp_error) / len(temp_error)
            x_data.append(precision_value)
            y_data.append(final_time_taken)
            y2_data.append(final_error)

        ax1.plot(x_data, y_data, label=method, linestyle="-")
        ax2.plot(x_data, y2_data, label=method, linestyle="--")
        x_data.clear()
        y_data.clear()
        y2_data.clear()

    # plotting of the data
    ax1.set_xlabel("Precision Value")
    ax1.set_ylabel("Execution Time (ms)")
    ax2.set_ylabel("Error")
    ax1.legend(bbox_to_anchor=(1, 1.3), loc="upper right", title="Method", fontsize="small")
    ax2.legend(bbox_to_anchor=(0, 1.3), loc="upper left", title="Error", fontsize="small")
    plt.tight_layout()
    plt.savefig("../graphs/sigmoid_function_approximations_benchmark.png")


# calling sigmoid_approximation_plot function
sigmoid_approximation_plot(benchmark_data_sigmoid)
