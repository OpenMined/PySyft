import logging


logger = logging.getLogger("run_websocket_server")
FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(level=logging.DEBUG)


import syft as sy
from syft.workers import WebsocketServerWorker
import torch
import argparse
from torchvision import datasets
from torchvision import transforms
import numpy as np

hook = sy.TorchHook(torch)


def start_server(participant, keep_labels=None, **kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    server = participant(**kwargs)

    mnist_trainset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    indices = np.isin(mnist_trainset.train_labels, keep_labels).astype("uint8")
    logger.info("nr true indices: %s", indices.sum())
    logger.info("train_data shape: %s", mnist_trainset.train_data.shape)
    selected_data = (
        torch.native_masked_select(mnist_trainset.train_data.transpose(0, 2), torch.tensor(indices))
        .view(28, 28, -1)
        .transpose(2, 0)
    )
    logger.info("after selection: %s", selected_data.shape)
    selected_targets = torch.native_masked_select(
        mnist_trainset.train_labels, torch.tensor(indices)
    )

    # logger.info("data: %s", mnist_trainset.data)
    # logger.info("target: %s", mnist_trainset.target)

    dataset = sy.BaseDataset(
        data=selected_data, targets=selected_targets, transform=mnist_trainset.transform
    )
    server.add_dataset(dataset, key="mnist")

    logger.info("datasets: %s", server.datasets)
    logger.info("len(datasets[mnist]): %s", len(server.datasets["mnist"]))

    server.start()
    return server


parser = argparse.ArgumentParser(description="Run websocket server worker.")
parser.add_argument(
    "--port", "-p", type=int, help="port number of the websocket server worker, e.g. --port 8777"
)
parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
parser.add_argument(
    "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
)
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="if set, websocket server worker will be started in verbose mode",
)

args = parser.parse_args()

# args.id = "alice"
# args.port = 8777

keep_labels_dict = {"alice": [0, 1, 2, 3], "bob": [4, 5, 6], "charlie": [7, 8, 9]}

kwargs = {
    "id": args.id,
    "host": args.host,
    "port": args.port,
    "hook": hook,
    "verbose": args.verbose,
    "keep_labels": keep_labels_dict[args.id],
}
server = start_server(WebsocketServerWorker, **kwargs)
