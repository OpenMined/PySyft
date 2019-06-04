import argparse
import logging

import torch as th
from syft.workers import WebsocketServerWorker

import syft as sy


def start_websocket_server(**kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    server = WebsocketServerWorker(**kwargs)

    # setup local data available on the websocket server
    data = th.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = th.tensor([[1], [0], [1], [0]], requires_grad=False)

    logger.info("data: %s", data)
    logger.info("target: %s", target)

    dataset = sy.BaseDataset(data, target)
    server.add_dataset(dataset, key="vectors")

    logger.info("datasets: %s", server.datasets)
    server.start()

    return server


if __name__ == "__main__":
    logger = logging.getLogger("run_websocket_server")
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.INFO)

    hook = sy.TorchHook(th)
    hook.local_worker = sy.VirtualWorker(
        id="local_worker_at_remote", is_client_worker=True, hook=hook
    )

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
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket server worker will be started in verbose mode",
    )

    args = parser.parse_args()

    kwargs = {
        "id": args.id,
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": args.verbose,
    }
    server = start_websocket_server(**kwargs)
