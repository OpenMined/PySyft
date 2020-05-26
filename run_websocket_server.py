from multiprocessing import Process
import argparse
import os
import logging
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from syft.frameworks.torch.fl import utils

KEEP_LABELS_DICT = {
    "alice": [0, 1, 2, 3],
    "bob": [4, 5, 6],
    "charlie": [7, 8, 9],
    "testing": list(range(10)),
    None: list(range(10)),
}


def start_websocket_server_worker(
    id, host, port, hook, verbose, keep_labels=None, training=True, pytest_testing=False
):
    """Helper function for spinning up a websocket server and setting up the local datasets."""

    server = WebsocketServerWorker(id=id, host=host, port=port, hook=hook, verbose=verbose)

    # Setup toy data (mnist example)
    mnist_dataset = datasets.MNIST(
        root="./data",
        train=training,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    if training:
        indices = np.isin(mnist_dataset.targets, keep_labels).astype("uint8")
        logger.info("number of true indices: %s", indices.sum())
        selected_data = (
            torch.native_masked_select(mnist_dataset.data.transpose(0, 2), torch.tensor(indices))
            .view(28, 28, -1)
            .transpose(2, 0)
        )
        logger.info("after selection: %s", selected_data.shape)
        selected_targets = torch.native_masked_select(mnist_dataset.targets, torch.tensor(indices))

        dataset = sy.BaseDataset(
            data=selected_data, targets=selected_targets, transform=mnist_dataset.transform
        )
        key = "mnist"
    else:
        dataset = sy.BaseDataset(
            data=mnist_dataset.data,
            targets=mnist_dataset.targets,
            transform=mnist_dataset.transform,
        )
        key = "mnist_testing"

    # Adding Dataset
    server.add_dataset(dataset, key=key)
    if pytest_testing:
        # Setup toy data (vectors example)
        data_vectors = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
        target_vectors = torch.tensor([[1], [0], [1], [0]])

        server.add_dataset(sy.BaseDataset(data_vectors, target_vectors), key="vectors")

        # Setup toy data (xor example)
        data_xor = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True
        )
        target_xor = torch.tensor([1.0, 1.0, 0.0, 0.0], requires_grad=False)

        server.add_dataset(sy.BaseDataset(data_xor, target_xor), key="xor")

        # Setup gaussian mixture dataset
        data, target = utils.create_gaussian_mixture_toy_data(nr_samples=100)
        server.add_dataset(sy.BaseDataset(data, target), key="gaussian_mixture")

        # Setup partial iris dataset
        data, target = utils.iris_data_partial()
        dataset = sy.BaseDataset(data, target)
        dataset_key = "iris"
        server.add_dataset(dataset, key=dataset_key)
    else:
        count = [0] * 10
        logger.info(
            "MNIST dataset (%s set), available numbers on %s: ", "train" if training else "test", id
        )
        for i in range(10):
            count[i] = (dataset.targets == i).sum().item()
            logger.info("      %s: %s", i, count[i])

    logger.info("datasets: %s", server.datasets)
    if training:
        logger.info("len(datasets[mnist]): %s", len(server.datasets[key]))

    server.start()
    return server


def start_proc(participant, kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)
        server.start()

    p = Process(target=target)
    p.start()
    return p


def start_proc_steal_data_over_sockets(participant, kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)
        private_data = torch.tensor([1, 1, 1, 1, 1])
        private_data.private = True
        server._objects[1] = private_data
        server.start()

    p = Process(target=target)
    p.start()
    return p


if __name__ == "__main__":

    # Logging setup
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d, p:%(process)d) - %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_server")
    logger.setLevel(level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help=(
            "if set, websocket server worker will load "
            "the test dataset instead of the training dataset"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""if set, websocket server worker will be started in verbose mode""",
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default="normal",
        help=(
            "can run websocket server for websockets examples of mnist/mnist-parallel or "
            "pen_testing/steal_data_over_sockets. Type 'mnist' for starting server "
            "for websockets-example-MNIST, `mnist-parallel` for websockets-example-MNIST-parallel "
            "and 'steal_data' for pen_tesing stealing data over sockets"
        ),
    )
    parser.add_argument("--pytest_testing", action="store_true", help="""Used for pytest testing""")
    args = parser.parse_args()

    # Hook and start server
    hook = sy.TorchHook(torch)

    # server = start_proc(WebsocketServerWorker, kwargs)
    if args.notebook == "normal" or args.notebook == "mnist" or args.notebook == "steal_data":
        kwargs = {
            "id": args.id,
            "host": args.host,
            "port": args.port,
            "hook": hook,
            "verbose": args.verbose,
        }
        if os.name != "nt" and (args.notebook == "normal" or args.notebook == "mnist"):
            server = start_proc(WebsocketServerWorker, kwargs)
        elif os.name != "nt" and args.notebook == "steal_data":
            server = start_proc_steal_data_over_sockets(WebsocketServerWorker, kwargs)
        else:
            server = WebsocketServerWorker(**kwargs)
            server.start()
    elif args.notebook == "mnist-parallel" or args.pytest_testing:
        server = start_websocket_server_worker(
            id=args.id,
            host=args.host,
            port=args.port,
            hook=hook,
            verbose=args.verbose,
            keep_labels=KEEP_LABELS_DICT[args.id]
            if args.id in KEEP_LABELS_DICT
            else list(range(10)),
            training=not args.testing,
        )
