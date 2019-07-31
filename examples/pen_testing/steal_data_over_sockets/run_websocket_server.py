from multiprocessing import Process
import syft as sy
from syft.workers import WebsocketServerWorker
import torch
import argparse
import os

hook = sy.TorchHook(torch)


def start_proc(participant, kwargs):  # pragma: no cover
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


if os.name != "nt":
    server = start_proc(WebsocketServerWorker, kwargs)
else:
    server = WebsocketServerWorker(**kwargs)
    server.start()
