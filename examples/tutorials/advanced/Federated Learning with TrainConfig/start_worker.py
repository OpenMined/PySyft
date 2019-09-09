import argparse

import torch as th
from syft.workers.websocket_server import WebsocketServerWorker

import syft as sy

# Arguments
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


def main(**kwargs):  # pragma: no cover
    """Helper function for spinning up a websocket participant."""

    # Create websocket worker
    worker = WebsocketServerWorker(**kwargs)

    # Setup toy data (xor example)
    data = th.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
    target = th.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)

    # Create a dataset using the toy data
    dataset = sy.BaseDataset(data, target)

    # Tell the worker about the dataset
    worker.add_dataset(dataset, key="xor")

    # Start worker
    worker.start()

    return worker


if __name__ == "__main__":
    hook = sy.TorchHook(th)

    args = parser.parse_args()
    kwargs = {
        "id": args.id,
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": args.verbose,
    }

    main(**kwargs)
