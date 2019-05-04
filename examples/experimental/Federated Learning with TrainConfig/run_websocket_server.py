import argparse
from multiprocessing import Process
import sys

import torch as th

import syft as sy
from syft.workers import WebsocketServerWorker

hook = sy.TorchHook(th)
hook.local_worker = sy.VirtualWorker(id="me, port: 8777", is_client_worker=True, hook=hook)


def start_proc(participant, kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)

        _ = th.tensor([[-1, 2.0], [0, 1.1]], id=("#data")).send(server)
        _ = th.tensor([[1.0], [0.0]], id=("#target")).send(server)

        # dataset = sy.BaseDataset(data, target_)
        # dataset.send(server)
        # server.dataset = dataset

        server.start()

    p = Process(target=target)
    p.start()
    return p


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

kwargs = {
    "id": args.id,
    "host": args.host,
    "port": args.port,
    "hook": hook,
    "verbose": args.verbose,
}
server = start_proc(WebsocketServerWorker, kwargs)
