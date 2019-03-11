import torch
import time
from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker
from multiprocessing import Process


def start_proc(participant, kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)
        server.start()

    p = Process(target=target)
    p.start()
    return p


def test_websocket_worker(hook):
    """Evaluates that you can do basic tensor operations using
    WebsocketServerWorker"""

    kwargs = {"id": "fed1", "host": "localhost", "port": 8765, "hook": hook}
    server = start_proc(WebsocketServerWorker, kwargs)

    time.sleep(0.1)
    x = torch.ones(5)

    socket_pipe = WebsocketClientWorker(**kwargs)

    x = x.send(socket_pipe)
    y = x + x
    y = y.get()

    assert (y == torch.ones(5) * 2).all()

    del x

    server.terminate()
