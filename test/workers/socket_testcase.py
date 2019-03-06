import torch
import syft as sy
import time
import asyncio
from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker
from multiprocessing import Process
from collections import ChainMap


hook = sy.TorchHook(torch)
ones = torch.ones(5)
ones.tags = ["ones"]


def start_proc(participant, kwargs):
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)
        # server.start()

    p = Process(target=target)
    p.start()
    return p


kwargs = {
    "verbose": True,
    "id": "fed1",
    "connection_params": {"host": "localhost", "port": 8765},
    "data": (ones),
    "hook": hook,
}
server = start_proc(WebsocketServerWorker, kwargs)


twos = torch.ones(5) + 0.5 + 0.5

time.sleep(1)
# connect bobby to the server:
socket_pipe = WebsocketClientWorker(**ChainMap({"id": f"bobby", "data": ()}, kwargs))
print("sending", twos)
twos_pointer = twos.send(socket_pipe)
print("foo")

ones_pointer = socket_pipe.search("ones")
print(ones_pointer)
#
##three = twos_pointer + ones_pointer
#
# assert three.get() == (torch.ones(5) + 2)
