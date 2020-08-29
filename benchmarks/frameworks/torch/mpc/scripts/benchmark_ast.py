import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import syft
import matplotlib.pyplot as plt

from benchmarks.frameworks.torch.mpc.scripts.workers_initialization import workers, hook
from benchmarks.frameworks.torch.mpc.scripts.benchmark_sample_data import (
    b_data_share_get,
    b_data_max_pool2d,
    b_data_avg_pool2d,
    b_data_batch_norm,
)


def benchmark_share_get(workers, protocol, dtype, n_workers):
    """
    This function benchmarks the share_get_functions.

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


def benchmark_max_pool2d(workers, protocol, prec_frac):
    """
    This function benchmarks max_plot2d function.

    Args:
        workers (dict): workers used for sharing data
        protocol (str): the name of the protocol
        prec_frac (int): the precision value (upper limit)
    """

    me, alice, bob, crypto_provider = (
        workers["me"],
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    args = (alice, bob)
    kwargs = dict(crypto_provider=crypto_provider, protocol=protocol)

    m = 4
    t = torch.tensor(list(range(3 * 7 * m * m))).float().reshape(3, 7, m, m)
    x = t.fix_prec(precision_fractional=prec_frac).share(*args, **kwargs)

    # using maxpool optimization for kernel_size=2
    expected = F.max_pool2d(t, kernel_size=2)
    result = F.max_pool2d(x, kernel_size=2).get().float_prec()

    # # without
    # expected = F.max_pool2d(t, kernel_size=3)
    # result = F.max_pool2d(x, kernel_size=3).get().float_prec()


def benchmark_max_pool2d_plot(b_data_max_pool2d):
    """
    This function plots the graph for various protocols benchmarks for
    max_pool2d.

    Args:
        b_data_max_pool2d (list): list of protocols to approximate

    Returns:
        benchmark_max_pool2d.png (png): plotted graph in graph/ast_benchmarks directory
    """

    # initializing workers
    worker = workers(hook())

    # getting data (protocols)
    protocols = b_data_max_pool2d

    # initializing graph plot
    fig, ax = plt.subplots()

    for protocol in protocols:

        # list for handling graph data
        x_data = []
        y_data = []

        for prec_frac in range(1, 5):
            temp_time_taken = []

            for i in range(10):
                start_time = timeit.default_timer()
                benchmark_max_pool2d(worker, protocol, prec_frac)
                time_taken = timeit.default_timer() - start_time
                temp_time_taken.append(time_taken)

            final_time_taken = sum(temp_time_taken) / len(temp_time_taken)
            final_time_taken *= 1000
            y_data.append(final_time_taken)
            x_data.append(prec_frac)

        ax.plot(x_data, y_data, label=protocol, linestyle="-")
        x_data.clear()
        y_data.clear()

    # plotting of the data
    plt.title("Benchmark max_pool2d")
    ax.set_xlabel("Precision Value")
    ax.set_ylabel("Execution Time (ms)")
    ax.legend(bbox_to_anchor=(1, 1.3), loc="upper right", title="Protocol", fontsize="small")
    plt.tight_layout()
    plt.savefig("../graphs/ast_benchmarks/benchmark_max_pool2d.png")


# calling benchmark_max_pool2d_plot
benchmark_max_pool2d_plot(b_data_max_pool2d)


def benchmark_avg_pool2d(workers, protocol, prec_frac):
    """
    This function benchmarks avg_plot2d function.

    Args:
        workers (dict): workers used for sharing data
        protocol (str): the name of the protocol
        prec_frac (int): the precision value (upper limit)
    """

    me, alice, bob, crypto_provider = (
        workers["me"],
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    args = (alice, bob)
    kwargs = dict(crypto_provider=crypto_provider, protocol=protocol)

    m = 4
    t = torch.tensor(list(range(3 * 7 * m * m))).float().reshape(3, 7, m, m)
    x = t.fix_prec(precision_fractional=prec_frac).share(*args, **kwargs)

    # using maxpool optimization for kernel_size=2
    expected = F.avg_pool2d(t, kernel_size=2)
    result = F.avg_pool2d(x, kernel_size=2).get().float_prec()

    # # without
    # expected = F.avg_pool2d(t, kernel_size=3)
    # result = F.avg_pool2d(x, kernel_size=3).get().float_prec()


def benchmark_avg_pool2d_plot(b_data_avg_pool2d):
    """
    This function plots the graph for various protocols benchmarks for
    avg_pool2d.

    Args:
        b_data_avg_pool2d (list): list of protocols to approximate

    Returns:
        benchmark_avg_pool2d.png (png): plotted graph in graph/ast_benchmarks directory
    """

    # initializing workers
    worker = workers(hook())

    # getting data (protocols)
    protocols = b_data_avg_pool2d

    # initializing graph plot
    fig, ax = plt.subplots()

    for protocol in protocols:

        # list for handling graph data
        x_data = []
        y_data = []

        for prec_frac in range(1, 5):
            temp_time_taken = []

            for i in range(10):
                start_time = timeit.default_timer()
                benchmark_avg_pool2d(worker, protocol, prec_frac)
                time_taken = timeit.default_timer() - start_time
                temp_time_taken.append(time_taken)

            final_time_taken = sum(temp_time_taken) / len(temp_time_taken)
            final_time_taken *= 1000
            y_data.append(final_time_taken)
            x_data.append(prec_frac)

        ax.plot(x_data, y_data, label=protocol, linestyle="-")
        x_data.clear()
        y_data.clear()

    # plotting of the data
    plt.title("Benchmark avg_pool2d")
    ax.set_xlabel("Precision Value")
    ax.set_ylabel("Execution Time (ms)")
    ax.legend(bbox_to_anchor=(1, 1.3), loc="upper right", title="Protocol", fontsize="small")
    plt.tight_layout()
    plt.savefig("../graphs/ast_benchmarks/benchmark_avg_pool2d.png")
    # plt.show()


# calling benchmark_avg_pool2d_plot
benchmark_avg_pool2d_plot(b_data_avg_pool2d)


def benchmark_batch_norm(workers, protocol, training, prec_frac):
    """
    This function benchmarks batch_norm function.

    Args:
        workers (dict): workers used for sharing data
        protocol (str): the name of the protocol
        training (bool): training or eval mode
        prec_frac (int): the precision value (upper limit)
    """

    me, alice, bob, crypto_provider = (
        workers["me"],
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    args = (alice, bob)
    syft.local_worker.clients = args
    kwargs = dict(crypto_provider=crypto_provider, protocol=protocol)

    model = nn.BatchNorm2d(4, momentum=0)
    if training:
        model.train()
    else:
        model.eval()

    x = torch.rand(1, 4, 5, 5)
    expected = model(x)

    model.fix_prec(precision_fractional=prec_frac).share(*args, **kwargs)
    x = x.fix_prec(precision_fractional=prec_frac).share(*args, **kwargs)
    y = model(x)
    predicted = y.get().float_prec()


def benchmark_batch_norm_plot(b_data_batch_norm):
    """
    This function plots the graph for various protocols benchmarks for
    batch_norm.

    Args:
        b_data_batch_norm (list): list of protocols to approximate

    Returns:
        benchmark_batch_norm.png (png): plotted graph in graph/ast_benchmarks directory
    """

    # initializing workers
    worker = workers(hook())

    # getting data (protocols)
    protocols = b_data_batch_norm

    # initializing graph plot
    fig, ax = plt.subplots()

    for protocol in protocols:

        # list for handling graph data
        x_data = []
        y_data = []

        for prec_frac in range(1, 5):
            temp_time_taken = []

            for i in range(10):
                start_time = timeit.default_timer()
                benchmark_batch_norm(worker, protocol, True, prec_frac)
                time_taken = timeit.default_timer() - start_time
                temp_time_taken.append(time_taken)

            final_time_taken = sum(temp_time_taken) / len(temp_time_taken)
            final_time_taken *= 1000
            y_data.append(final_time_taken)
            x_data.append(prec_frac)

        ax.plot(x_data, y_data, label=protocol, linestyle="-")
        x_data.clear()
        y_data.clear()

    # plotting of the data
    plt.title("benchmark_batch_norm")
    ax.set_xlabel("Precision Value")
    ax.set_ylabel("Execution Time (ms)")
    ax.legend(bbox_to_anchor=(1, 1.3), loc="upper right", title="Protocol", fontsize="small")
    plt.tight_layout()
    plt.savefig("../graphs/ast_benchmarks/benchmark_batch_norm.png")
    # plt.show()


# calling benchmark_batch_norm_plot
benchmark_batch_norm_plot(b_data_batch_norm)
