import argparse
from multiprocessing import Process
import logging

import torch as th

import syft as sy
from syft.frameworks.torch.tensors.interpreters import AutogradTensor
from syft.workers import WebsocketServerWorker

logger = logging.getLogger("run_websocket_server")
FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(level=logging.INFO)

hook = sy.TorchHook(th)
hook.local_worker = sy.VirtualWorker(id="local_worker_at_remote", is_client_worker=True, hook=hook)


def start_proc(participant, kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)
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
